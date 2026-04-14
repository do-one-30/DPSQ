import torch
import torch.nn as nn
from datasets import load_dataset
import evaluate
from transformers import AutoImageProcessor
from applications.custom_quant import CustomLinear, CustomLinearCalib, CustomLinearEval

def replace_vision_mlp2_layer(model: nn.Module, method_func, tile_m: int, tile_n: int, tile_k: int, **kwargs):
    replaced_count = 0
    for name, module in model.named_modules():
        if name.endswith("mlp.dense2") or name.endswith("mlp.fc2") or name.endswith("ffn.fc2"):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            old_fc2 = getattr(parent_module, child_name)

            if isinstance(old_fc2, nn.Linear):
                new_fc2 = CustomLinear(
                    in_features=old_fc2.in_features,
                    out_features=old_fc2.out_features,
                    method_func=method_func,
                    bias=(old_fc2.bias is not None),
                    tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                    **kwargs
                ).to(old_fc2.weight.device, dtype=old_fc2.weight.dtype)

                with torch.no_grad():
                    new_fc2.weight.copy_(old_fc2.weight)
                    if old_fc2.bias is not None: 
                        new_fc2.bias.copy_(old_fc2.bias)

                setattr(parent_module, child_name, new_fc2)
                replaced_count += 1
                
    return model

def replace_vision_mlp2_layer_for_calib(model: nn.Module, calib_method, tile_m: int, tile_n: int, tile_k: int, gs: int, eps: float):
    replaced_count = 0
    for name, module in model.named_modules():
        if name.endswith("mlp.dense2") or name.endswith("mlp.fc2") or name.endswith("ffn.fc2"):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            old_fc2 = getattr(parent_module, child_name)

            if isinstance(old_fc2, nn.Linear):
                new_fc2 = CustomLinearCalib(
                    in_features=old_fc2.in_features, out_features=old_fc2.out_features,
                    calib_method=calib_method, bias=(old_fc2.bias is not None),
                    tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps,
                ).to(old_fc2.weight.device, dtype=old_fc2.weight.dtype)

                with torch.no_grad():
                    new_fc2.weight.copy_(old_fc2.weight)
                    if old_fc2.bias is not None: 
                        new_fc2.bias.copy_(old_fc2.bias)

                setattr(parent_module, child_name, new_fc2)
                replaced_count += 1
                
    return model

def replace_vision_mlp2_layer_for_eval(model: nn.Module, apply_method, layer_scales: dict, tile_m: int, tile_n: int, tile_k: int, gs: int, eps: float):
    replaced_count = 0
    for name, module in model.named_modules():
        if name.endswith("mlp.dense2") or name.endswith("mlp.fc2") or name.endswith("ffn.fc2"):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            old_fc2 = getattr(parent_module, child_name)

            if isinstance(old_fc2, nn.Linear):
                if name not in layer_scales:
                    raise ValueError(f"Scale for {name} not found in loaded scales!")
                step_scales = layer_scales[name]

                new_fc2 = CustomLinearEval(
                    in_features=old_fc2.in_features, out_features=old_fc2.out_features,
                    apply_method=apply_method, step_scales=step_scales, bias=(old_fc2.bias is not None),
                    tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps,
                ).to(old_fc2.weight.device, dtype=old_fc2.weight.dtype)

                with torch.no_grad():
                    new_fc2.weight.copy_(old_fc2.weight)
                    if old_fc2.bias is not None: 
                        new_fc2.bias.copy_(old_fc2.bias)

                setattr(parent_module, child_name, new_fc2)
                replaced_count += 1
                
    return model


def prepare_seg_dataset_and_metrics(model_name_or_path: str, num_labels: int, batch_size: int):
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    image_processor.do_reduce_labels = True
    
    dataset = load_dataset("scene_parse_150", trust_remote_code=True)

    def preprocess_batch(example_batch):
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = image_processor(images, segmentation_maps=labels)
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": inputs["labels"],
        }

    eval_dataset = dataset["validation"].map(
        preprocess_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["validation"].column_names,
    )

    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            if isinstance(logits, tuple):
                logits = logits[0]
                
            logits_tensor = torch.from_numpy(logits)
            upsampled_logits = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            
            predicted_labels = upsampled_logits.argmax(dim=1).numpy()
            metrics = metric.compute(
                predictions=predicted_labels,
                references=labels,
                num_labels=num_labels,
                ignore_index=255,
                reduce_labels=False
            )
            return {
                "mean_iou": metrics["mean_iou"],
                "mean_accuracy": metrics["mean_accuracy"],
            }

    def data_collator(features):
        pixel_values = [torch.tensor(f["pixel_values"]) if not isinstance(f["pixel_values"], torch.Tensor)
                        else f["pixel_values"] for f in features]
        labels = [torch.tensor(f["labels"]) if not isinstance(f["labels"], torch.Tensor)
                else f["labels"] for f in features]
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.stack(labels),
        }

    return eval_dataset, compute_metrics, data_collator



def prepare_seg_datasets_and_utils(model_name_or_path: str, num_labels: int):
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    image_processor.do_reduce_labels = True
    
    dataset = load_dataset("scene_parse_150", trust_remote_code=True)

    def preprocess_batch(example_batch):
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = image_processor(images, segmentation_maps=labels)
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": inputs["labels"],
        }

    def data_collator(features):
        pixel_values = [torch.tensor(f["pixel_values"]) if not isinstance(f["pixel_values"], torch.Tensor)
                        else f["pixel_values"] for f in features]
        labels = [torch.tensor(f["labels"]) if not isinstance(f["labels"], torch.Tensor)
                else f["labels"] for f in features]
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.stack(labels),
        }

    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            if isinstance(logits, tuple): logits = logits[0]
                
            logits_tensor = torch.from_numpy(logits)
            upsampled_logits = nn.functional.interpolate(
                logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            
            predicted_labels = upsampled_logits.argmax(dim=1).numpy()
            metrics = metric.compute(
                predictions=predicted_labels, references=labels,
                num_labels=num_labels, ignore_index=255, reduce_labels=False
            )
            return {"mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

    return dataset, preprocess_batch, data_collator, compute_metrics