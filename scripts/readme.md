## Batch runners for the PSUM-level baselines (LLM_int8 / OWQ)

Two convenience scripts sweep both baselines over all downstream tasks and
bit-widths. Run them with `bash` (not tcsh). They `cd` to the repo root
themselves and put results under `$OUTDIR/<method>/` (per-method dirs, since the
BERT eval's result filename does not include the method name). For OWQ they run
the calibration pass and the evaluation back-to-back.

```bash
# BERT-Base GLUE: cola mnli mrpc qnli rte stsb  (default bits "8 4")
bash scripts/run_baselines_bert.sh

# LLaMA-2-7B zero-shot: boolq piqa hellaswag winog arc-e arc-c obqa
bash scripts/run_baselines_llama.sh
```

Both accept overridable env vars (defaults in brackets): `PYTHON[python]`,
`GPU[0]`, `TILE[128]`, `BITS[8 4]`, `OUTDIR[./eval_output]`, `CALIB_RATIO[0.1]`,
`OUTLIER_PER_TILE[1.0]`, `BATCH[16]`, `METHODS[LLM_int8 OWQ]`, `TASKS[...]`
(plus `MODEL[meta-llama/Llama-2-7b-hf]` for LLaMA). Examples:

```bash
BITS="8 6 4" GPU=1 bash scripts/run_baselines_bert.sh          # all bit-widths
TASKS="mrpc rte" METHODS="OWQ" bash scripts/run_baselines_bert.sh   # subset
TASKS="obqa piqa" METHODS="LLM_int8" bash scripts/run_baselines_llama.sh
```

`OUTLIER_PER_TILE` is the shared per-tile FP32 budget for both baselines (and
matches DPSQ's `target_outlier_per_tile`), so all three methods are compared at
the same high-precision budget. Each baseline's result JSON also records
`hp_fraction` (average fraction of PSUM columns kept in FP32).

The commands each runner issues are documented individually in the per-method
sections below.

## Common options

All methods now accept `--bits {8,6,4}`. The default is `--bits 8`, which preserves the previous INT8 setting. For method B/C, use the same `--bits` value for both calibration and evaluation because calibrated step scales are bitwidth-dependent.

DPSQ alpha can be calibrated on a calibration subset by adding `--calibrate_alpha` to the ADEF/DPSQ evaluation script. The calibration run writes a `.pt` alpha file and a CSV with one row per candidate alpha. The selected alpha is the first candidate whose `outlier/tile` is lower than `--target_outlier_per_tile` (default: 1.0).

For the Method G experiment script, calibration is run once per dataset with `CALIB_RATIO=0.1` by default, then one global alpha is selected per model group and bitwidth. Each dataset first selects the minimum alpha where `outlier/tile < TARGET_OUTLIER_PER_TILE`; the global alpha is the maximum of those per-dataset minimum alphas. BERT evaluation reuses `alpha_calib/methodG/bert/global/bits*/global_alpha.pt`, and LLaMA evaluation reuses `alpha_calib/methodG/llama/global/bits*/global_alpha.pt`.

Example for BERT DPSQ INT4 alpha calibration:
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name DPSQ --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --bits 4 --calibrate_alpha --calib_ratio 0.1 --alpha_candidates 0.5:12.0:0.5 --alpha_save_path ./eval_output/dpsq_alpha_mrpc_bits4.pt --alpha_csv_path ./eval_output/dpsq_alpha_mrpc_bits4.csv
```

Then evaluate with the calibrated alpha:
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name DPSQ --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --bits 4 --alpha_path ./eval_output/dpsq_alpha_mrpc_bits4.pt
```

The same alpha calibration options are available for LLaMA, SegFormer, and EfficientViT ADEF/DPSQ scripts. For EfficientViT alpha calibration, pass the calibration image path through `--path`; for evaluation, pass the validation image path and reuse `--alpha_path`.

## BERT_Base

### method A
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodA --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128
```

### method B, C (Change the value of gs to 1, 2, and 4.)
```bash
# calib
python -m applications.BERT_Base.eval_BERT_BC calib --method_file ./methods/methods.py --method_name methodB_C_calib --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --gs 1
```
```bash
# eval
python -m applications.BERT_Base.eval_BERT_BC eval --method_file ./methods/methods.py --method_name methodB_C --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --gs 1
```

### method D
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodD --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128
```

### method E
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodE --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128
```

### method F
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodF --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128
```

### DPSQ (method G in our paper)
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name DPSQ --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128
```

### LLM_int8 (PSUM-level, dynamic, approach A)
No threshold and no calibration. `--llm_int8_outlier_per_tile` sets the average number of FP32 columns per PSUM tile (default 1.0, matching DPSQ's `target_outlier_per_tile`); internally it keeps the top-r columns by magnitude with `r = round(outlier_per_tile · ⌈K/tile_k⌉)`. The result JSON reports `hp_fraction` (average fraction of PSUM columns kept in FP32).
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name LLM_int8 --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --bits 8 --llm_int8_outlier_per_tile 1.0
```

### OWQ (PSUM-level, calibration-based, approach A)
OWQ is a two-stage flow like method B/C. First calibrate the weak columns on a subset of the training set (`--calib_ratio 0.1` by default); `--owq_outlier_per_tile` sets the average number of FP32 weak columns per PSUM tile (default 1.0, same budget notion as LLM_int8/DPSQ). Use the **same `--bits`** for calibration and evaluation.
```bash
# calib (selects weak columns, saves a .pt, then exits)
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name OWQ --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --bits 8 --calibrate_owq --calib_ratio 0.1 --owq_outlier_per_tile 1.0 --owq_calib_path ./eval_output/owq_mrpc_bits8.pt
```
```bash
# eval (reuses the calibrated weak columns; reports hp_fraction)
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name OWQ --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --bits 8 --owq_calib_path ./eval_output/owq_mrpc_bits8.pt
```

## LLaMA2-7B

### method A
```bash
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodA --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128
```

### method B, C (Change the value of gs to 1, 2, and 4.)
```bash
# calib
python -m applications.LLAMA_2_7B.eval_LLAMA_BC calib --method_file ./methods/methods.py --method_name methodB_C_calib --dataset_name obqa --tile_size 128 --gs 1
```

```bash
#eval
python -m applications.LLAMA_2_7B.eval_LLAMA_BC eval --method_file ./methods/methods.py --method_name methodB_C --dataset_name obqa --tile_size 128 --gs 1
```

### method D
```bash
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodD --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128
```

### method E
```bash
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodE --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128
```

### method F
```bash
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodF --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128
```

### DPSQ (method G in our paper)
```bash
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name DPSQ --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128 --alpha 9.9
```

### LLM_int8 (PSUM-level, dynamic, approach A)
```bash
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name LLM_int8 --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128 --bits 8 --llm_int8_outlier_per_tile 1.0
```

### OWQ (PSUM-level, calibration-based, approach A)
```bash
# calib
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name OWQ --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128 --bits 8 --calibrate_owq --calib_ratio 0.1 --owq_outlier_per_tile 1.0 --owq_calib_path ./eval_output/owq_obqa_bits8.pt
```
```bash
# eval
python -m applications.LLAMA_2_7B.eval_LLAMA_ADEF_DPSQ --method_file ./methods/methods.py --method_name OWQ --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name obqa --tile_size 128 --bits 8 --owq_calib_path ./eval_output/owq_obqa_bits8.pt
```

## SegFormer-B0

### method A
```bash
python -m applications.SegFormer.eval_SegFormer_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodA --model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512 --tile_size 128
```

### method B, C (Change the value of gs to 1, 2, and 4.)
```bash
# calib
python -m applications.SegFormer.eval_SegFormer_BC calib --method_file ./methods/methods.py --method_name methodB_C_calib --model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512 --tile_size 128 --gs 1
```

```bash
# eval
python -m applications.SegFormer.eval_SegFormer_BC eval --method_file ./methods/methods.py --method_name methodB_C --model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512 --tile_size 128 --gs 1 
```

### method D
```bash
python -m applications.SegFormer.eval_SegFormer_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodD --model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512 --tile_size 128
```

### method E
```bash
python -m applications.SegFormer.eval_SegFormer_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodE --model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512 --tile_size 128
```

### method F
```bash
python -m applications.SegFormer.eval_SegFormer_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodF --model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512 --tile_size 128
```

### DPSQ (method G in our paper)
```bash
python -m applications.SegFormer.eval_SegFormer_ADEF_DPSQ --method_file ./methods/methods.py --method_name DPSQ --model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512 --tile_size 128
```


## EfficientViT-B1


### method A
```bash
python -m applications.EfficientViT.eval_EVIT_ADEF_DPSQ --dataset ade20k --path ./scene_parse_export/images/validation --model efficientvit-seg-b1-ade20k --weight_url ./efficientvit/assets/checkpoints/efficientvit_seg_b1_ade20k.pt -j 16 --crop_size 512 --save_path ./methodA --method_file ./methods/methods.py --method_name methodA --tile_size 128
```

### method B, C (Change the value of gs to 1, 2, and 4.)
```bash
# calib
python -m applications.EfficientViT.eval_EVIT_B_C calib --dataset ade20k --path ./scene_parse_export/images/train --model efficientvit-seg-b1-ade20k --weight_url ./efficientvit/assets/checkpoints/efficientvit_seg_b1_ade20k.pt -j 16 --crop_size 512 --method_file ./methods/methods.py --method_name methodB_C_calib --tile_size 128 --gs 1
```
```bash
# eval
python -m applications.EfficientViT.eval_EVIT_B_C eval --dataset ade20k --path ./scene_parse_export/images/validation --model efficientvit-seg-b1-ade20k --weight_url ./efficientvit/assets/checkpoints/efficientvit_seg_b1_ade20k.pt -j 16 --crop_size 512 --method_file ./methods/methods.py --method_name methodB_C --tile_size 128 --gs 1
```

### method D
```bash
python -m applications.EfficientViT.eval_EVIT_ADEF_DPSQ --dataset ade20k --path ./scene_parse_export/images/validation --model efficientvit-seg-b1-ade20k --weight_url ./efficientvit/assets/checkpoints/efficientvit_seg_b1_ade20k.pt -j 16 --crop_size 512 --save_path ./methodA --method_file ./methods/methods.py --method_name methodD --tile_size 128
```

### method E
```bash
python -m applications.EfficientViT.eval_EVIT_ADEF_DPSQ --dataset ade20k --path ./scene_parse_export/images/validation --model efficientvit-seg-b1-ade20k --weight_url ./efficientvit/assets/checkpoints/efficientvit_seg_b1_ade20k.pt -j 16 --crop_size 512 --save_path ./methodA --method_file ./methods/methods.py --method_name methodE --tile_size 128
```

### method F
```bash
python -m applications.EfficientViT.eval_EVIT_ADEF_DPSQ --dataset ade20k --path ./scene_parse_export/images/validation --model efficientvit-seg-b1-ade20k --weight_url ./efficientvit/assets/checkpoints/efficientvit_seg_b1_ade20k.pt -j 16 --crop_size 512 --save_path ./methodA --method_file ./methods/methods.py --method_name methodF --tile_size 128
```

### DPSQ (method G in our paper)
```bash
python -m applications.EfficientViT.eval_EVIT_ADEF_DPSQ --dataset ade20k --path ./scene_parse_export/images/validation --model efficientvit-seg-b1-ade20k --weight_url ./efficientvit/assets/checkpoints/efficientvit_seg_b1_ade20k.pt -j 16 --crop_size 512 --save_path ./methodA --method_file ./methods/methods.py --method_name DPSQ --tile_size 128
```

