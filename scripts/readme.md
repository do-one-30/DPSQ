## BERT_Base

### method A
```bash
python -m applications.BERT_Base.eval_BERT_ADEF_DPSQ --method_file ./methods/methods.py --method_name methodA --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128
```

### method B
```bash
# calib
python -m applications.BERT_Base.eval_BERT_BC calib --method_file ./methods/methods.py --method_name methodB_C_calib --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --gs 1
# eval
python -m applications.BERT_Base.eval_BERT_BC eval --method_file ./methods/methods.py --method_name methodB_C --model_name_or_path textattack/bert-base-uncased-MRPC --dataset_name mrpc --tile_size 128 --gs 1
```

## LLaMA2-7B

## SegFormer-B0

## EfficientViT-B1