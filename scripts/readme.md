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

