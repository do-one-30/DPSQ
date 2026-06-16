import argparse
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#efficientvit_path = os.path.join(BASE_DIR, "efficientvit")
#if os.path.exists(efficientvit_path):
    #sys.path.append(BASE_DIR)
#else:
    #print("[Warning] 'Cannot find efficientvit'")

from efficientvit.efficientvit.apps.utils import AverageMeter
from efficientvit.efficientvit.models.utils import resize
from efficientvit.efficientvit.seg_model_zoo import create_efficientvit_seg_model

from applications.custom_quant import (
    load_method_from_file,
    CustomPointConv,
    DPSQAlphaStats,
    load_alpha_calibration,
    parse_alpha_candidates,
    save_alpha_calibration,
    select_alpha_row,
    update_custom_method_kwargs,
    write_alpha_calibration_csv,
)
from .data_utils_EVIT import (
    CityscapesDataset, ADE20KDataset, SegIOU, get_canvas,
    replace_efficientvit_1x1_conv
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom Quantization Methods on EfficientViT")
    
    parser.add_argument("--dataset", type=str, default="ade20k", choices=["cityscapes", "ade20k"])
    parser.add_argument("--path", type=str, required=True, help="Validation 데이터셋 경로")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    
    parser.add_argument("--method_file", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--save_path", type=str, default=None)
    
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--alpha_path", type=str, default=None)
    parser.add_argument("--calibrate_alpha", action="store_true")
    parser.add_argument("--alpha_candidates", type=str, default="0.5:12.0:0.5")
    parser.add_argument("--target_outlier_per_tile", type=float, default=1.0)
    parser.add_argument("--alpha_csv_path", type=str, default=None)
    parser.add_argument("--alpha_save_path", type=str, default=None)
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1)


    args.path = os.path.expanduser(args.path)
    if args.dataset == "cityscapes":
        dataset = CityscapesDataset(args.path, (args.crop_size, args.crop_size * 2))
    elif args.dataset == "ade20k":
        dataset = ADE20KDataset(args.path, crop_size=args.crop_size)
        
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    model = create_efficientvit_seg_model(args.model, weight_url=args.weight_url)
    selected_method = load_method_from_file(args.method_file, args.method_name)

    alpha_scale = args.alpha
    if args.alpha_path is not None and not args.calibrate_alpha:
        alpha_scale, _ = load_alpha_calibration(args.alpha_path, bits=args.bits)

    model, _ = replace_efficientvit_1x1_conv(
        model, selected_method, tile_m=args.tile_size, tile_n=args.tile_size,
        tile_k=args.tile_size, alpha_scale=alpha_scale, bits=args.bits
    )

    if args.calibrate_alpha:
        if args.method_name != "DPSQ":
            raise ValueError("--calibrate_alpha is only supported for method_name DPSQ")

        torch.manual_seed(args.seed)
        model = model.cuda().eval()
        calib_size = max(1, int(len(dataset) * args.calib_ratio))
        calib_size = min(calib_size, len(dataset))
        indices = torch.randperm(len(dataset))[:calib_size].tolist()
        calib_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, indices), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
        )

        rows = []
        candidates = parse_alpha_candidates(args.alpha_candidates)
        for alpha in candidates:
            stats = DPSQAlphaStats()
            update_custom_method_kwargs(
                model, CustomPointConv, alpha_scale=alpha, bits=args.bits, alpha_stats=stats
            )
            with torch.inference_mode():
                for feed_dict in tqdm(calib_loader, desc=f"Alpha {alpha:.4f}"):
                    images = feed_dict["data"].cuda()
                    _ = model(images)

            summary = stats.summary()
            row = {
                "alpha_scale": alpha,
                "bits": args.bits,
                "outliers": summary["outliers"],
                "tiles": summary["tiles"],
                "outlier_per_tile": summary["outlier_per_tile"],
                "target_outlier_per_tile": args.target_outlier_per_tile,
                "selected": False,
            }
            rows.append(row)
            print(f"  alpha={alpha:.4f}, outlier/tile={row['outlier_per_tile']:.6f}")

        best_row = select_alpha_row(rows, args.target_outlier_per_tile)
        for row in rows:
            row["selected"] = row is best_row

        csv_path = args.alpha_csv_path or os.path.join(
            args.output_dir, f"dpsq_alpha_calib_{args.dataset}_{args.model}_bits{args.bits}.csv"
        )
        save_path = args.alpha_save_path or os.path.join(
            args.output_dir, f"dpsq_alpha_{args.dataset}_{args.model}_bits{args.bits}.pt"
        )
        result = {
            "dataset": args.dataset,
            "model": args.model,
            "method_name": args.method_name,
            "tile_size": args.tile_size,
            "bits": args.bits,
            "calib_ratio": args.calib_ratio,
            "target_outlier_per_tile": args.target_outlier_per_tile,
            "alpha_candidates": candidates,
            "alpha_scale": best_row["alpha_scale"],
            "outlier_per_tile": best_row["outlier_per_tile"],
            "rows": rows,
        }
        write_alpha_calibration_csv(csv_path, rows)
        save_alpha_calibration(save_path, result)
        print(f"\n[*] Selected alpha={best_row['alpha_scale']:.4f}; saved {save_path}")
        print(f"[*] Wrote alpha calibration CSV: {csv_path}")
        return

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    interaction = AverageMeter(is_distributed=False)
    union = AverageMeter(is_distributed=False)
    iou = SegIOU(len(dataset.classes))
    
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)

    print("\n================ Evaluation Start ================")
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model}") as t:
            for feed_dict in data_loader:
                images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
                output = model(images)
                
                if output.shape[-2:] != mask.shape[-2:]:
                    output = resize(output, size=mask.shape[-2:])
                output = torch.argmax(output, dim=1)
                
                stats = iou(output, mask)
                interaction.update(stats["i"])
                union.update(stats["u"])

                t.set_postfix({"mIOU": (interaction.sum / union.sum).cpu().mean().item() * 100})
                t.update()

                if args.save_path is not None:
                    with open(os.path.join(args.save_path, "summary.txt"), "a") as fout:
                        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
                            pred = output[i].cpu().numpy()
                            from PIL import Image
                            raw_image = np.array(Image.open(image_path).convert("RGB"))
                            canvas = get_canvas(raw_image, pred, dataset.class_colors)
                            Image.fromarray(canvas).save(os.path.join(args.save_path, f"{idx}.png"))
                            fout.write(f"{idx}:\t{image_path}\n")

    final_miou = (interaction.sum / union.sum).cpu().mean().item() * 100
    os.makedirs(args.output_dir, exist_ok=True) 

    alpha_str = f"_a{alpha_scale}" if alpha_scale is not None else ""
    output_file = os.path.join(args.output_dir, f"evit_results_{args.dataset}_{args.method_name}_bits{args.bits}{alpha_str}.json")

    result_data = {
        "dataset": args.dataset,
        "model": args.model,
        "method_name": args.method_name,
        "kwargs_used": alpha_scale,
        "bits": args.bits,
        "mean_iou": final_miou
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)
        
    print(f"\n[Final Results - {args.dataset.upper()}]")
    print(f"  > mIoU: {final_miou:.3f}")
    print("==================================================\n")

if __name__ == "__main__":
    main()