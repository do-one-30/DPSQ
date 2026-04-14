import argparse
import os
import sys
import json
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


from efficientvit.efficientvit.apps.utils import AverageMeter
from efficientvit.efficientvit.models.utils import resize
from efficientvit.efficientvit.seg_model_zoo import create_efficientvit_seg_model

from applications.custom_quant import load_method_from_file, collect_evit_layer_mean_scales
from .data_utils_EVIT import (
    ADE20KDataset, CityscapesDataset, SegIOU, get_canvas,
    replace_efficientvit_1x1_conv_for_calib,
    replace_efficientvit_1x1_conv_for_eval,
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientViT Method C (Calib/Eval)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", type=str, default="ade20k", choices=["cityscapes", "ade20k"])
    common.add_argument("--path", type=str, required=True)
    common.add_argument("--model", type=str, required=True)
    common.add_argument("--weight_url", type=str, default=None)
    common.add_argument("-j", "--workers", type=int, default=4)
    common.add_argument("--crop_size", type=int, default=512)

    common.add_argument("--method_file", type=str, required=True)
    common.add_argument("--method_name", type=str, required=True)
    common.add_argument("--tile_size", type=int, default=16)
    common.add_argument("--gs", type=int, default=1)
    common.add_argument("--batch_size", type=int, default=1)
    common.add_argument("--gpu", type=str, default="0")
    common.add_argument("--eps", type=float, default=1e-8)

    calib_p = subparsers.add_parser("calib", parents=[common])
    calib_p.add_argument("--save_path", type=str, default="./calib_scales.pth")
    calib_p.add_argument("--calib_ratio", type=float, default=0.1)
    calib_p.add_argument("--seed", type=int, default=42)

    eval_p = subparsers.add_parser("eval", parents=[common])
    eval_p.add_argument("--scale_path", type=str, default="./calib_scales.pth")
    eval_p.add_argument("--output_dir", type=str, default="./eval_output")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dataset == "cityscapes":
        dataset = CityscapesDataset(os.path.expanduser(args.path), (512, 1024))
    else:
        dataset = ADE20KDataset(os.path.expanduser(args.path), crop_size=args.crop_size)

    model = create_efficientvit_seg_model(args.model, weight_url=args.weight_url)
    selected_method = load_method_from_file(args.method_file, args.method_name)

    if args.mode == "calib":
        torch.manual_seed(args.seed)
        model = replace_efficientvit_1x1_conv_for_calib(model, selected_method, args.tile_size, args.gs, 1e-8)
        model = torch.nn.DataParallel(model).cuda().eval()

        calib_size = max(1, int(len(dataset) * args.calib_ratio))
        indices = torch.randperm(len(dataset))[:calib_size].tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers)

        print("[*] Calibration Forward Pass...")
        with torch.no_grad():
            for batch in tqdm(loader):
                model(batch["data"].cuda())

        scales = collect_evit_layer_mean_scales(model.module)
        torch.save({"layer_scales": {k: v.cpu() for k, v in scales.items()}}, args.save_path)

    elif args.mode == "eval":
        scales = torch.load(args.scale_path, map_location="cpu")["layer_scales"]
        model = replace_efficientvit_1x1_conv_for_eval(model, selected_method, scales, args.tile_size, args.gs, 1e-8)
        model = torch.nn.DataParallel(model).cuda().eval()

        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
        iou = SegIOU(len(dataset.classes))
        inter, union = AverageMeter(False), AverageMeter(False)

        print("[*] Evaluation Start...")
        with torch.inference_mode():
            for batch in tqdm(loader):
                imgs, mask = batch["data"].cuda(), batch["label"].cuda()
                out = model(imgs)
                out = torch.argmax(resize(out, size=mask.shape[-2:]), dim=1)
                
                stats = iou(out, mask)
                inter.update(stats["i"]); union.update(stats["u"])

        miou = (inter.sum / union.sum).cpu().mean().item() * 100
        print(f"\n[Result] mIoU: {miou:.3f}")

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"evit_res_C_{args.method_name}.json"), "w") as f:
            json.dump({"mean_iou": miou}, f, indent=4)

if __name__ == "__main__":
    main()