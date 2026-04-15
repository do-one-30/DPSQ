# DPSQ

Official PyTorch implementation of **DPSQ: Dynamic Partial Sum Quantization with Mixed Granularity**.

This repository provides the reference implementation of DPSQ and several baseline PSUM quantization methods under a unified tiled INT8 simulation framework. The codebase includes evaluation pipelines for language and vision models, covering BERT-Base, LLaMA-2-7B, SegFormer-B0, and EfficientViT-B1.

---

## Overview

Partial-sum (PSUM) quantization is an important technique for reducing memory traffic and improving hardware efficiency in quantized neural network accelerators. However, conventional PSUM quantization often suffers from substantial accuracy degradation due to the large dynamic range variation of intermediate accumulations.

DPSQ addresses this problem with a mixed-granularity dynamic quantization strategy that selectively applies finer-grained scaling to outlier rows while preserving coarser-grained scaling for the remaining rows. This repository contains:

- the implementation of DPSQ,
- several comparison methods under the same tiled computation framework,
- reusable quantized layer wrappers for different model families,
- evaluation pipelines for NLP and vision benchmarks.

---

## Implemented Methods

The following methods are implemented in `methods/methods.py`:

- `methodA`: no PSUM quantization
- `methodB_C_calib`: calibration phase for precomputed step-scale methods
- `methodB_C`: evaluation phase using calibrated step scales
- `methodD`: tile-wise dynamic PSUM quantization
- `methodE`: row-wise dynamic PSUM quantization
- `methodF`: column-wise dynamic PSUM quantization
- `DPSQ`: proposed mixed-granularity dynamic PSUM quantization

---

## Repository Structure

```text
DPSQ/
├── applications/
│   ├── custom_quant.py
│   ├── BERT_Base/
│   │   ├── data_utils_BERT_Base.py
│   │   ├── eval_BERT_ADEF_DPSQ.py
│   │   └── eval_BERT_BC.py
│   ├── LLAMA_2_7B/
│   │   ├── data_utils_LLAMA.py
│   │   ├── eval_LLAMA_ADEF_DPSQ.py
│   │   └── eval_LLAMA_BC.py
│   ├── SegFormer/
│   │   ├── data_utils_SegFormer.py
│   │   ├── eval_SegFormer_ADEF_DPSQ.py
│   │   └── eval_SegFormer_BC.py
│   └── EfficientViT/
│       ├── data_utils_EVIT.py
│       ├── eval_EVIT_ADEF_DPSQ.py
│       └── eval_EVIT_B_C.py
├── methods/
│   ├── methods.py
│   └── methods_utils.py
└── .gitmodules
```

---

## Supported Models and Tasks

All methods are evaluated under a unified tiled INT8 framework. In all experiments, **input tiles are quantized row-wise, and weight tiles are quantized column-wise**. The compared methods differ only in how partial sums are quantized during accumulation.


### Language Models
* BERT_Base
    * GLUE tasks:
        * CoLA
        * MNLI
        * MRPC
        * QNLI
        * RTE
        * STS-B
    
* LLaMA-2-7B
    * Zero-shot common sense reasoging tasks:
        * BoolQ
        * PIQA
        * HellaSwag
        * Winogrande
        * ARC-Easy
        * ARC-Challenge
        * OpenBookQA

* Vision Models
    * SegFormer-B0
        * ADE20K semantic segmentation
    * EfficientViT-B1
        * ADE20K semantic segmentation

---

## Pretrained Models

The following Hugging Face checkpoints are used in this repository.

### Hugging Face Model Links

| Model Family | Task / Usage | Hugging Face Checkpoint |
|---|---|---|
| BERT-Base | CoLA | [textattack/bert-base-uncased-CoLA](https://huggingface.co/textattack/bert-base-uncased-CoLA) |
| BERT-Base | MNLI | [textattack/bert-base-uncased-MNLI](https://huggingface.co/textattack/bert-base-uncased-MNLI) |
| BERT-Base | MRPC | [textattack/bert-base-uncased-MRPC](https://huggingface.co/textattack/bert-base-uncased-MRPC) |
| BERT-Base | QNLI | [textattack/bert-base-uncased-QNLI](https://huggingface.co/textattack/bert-base-uncased-QNLI) |
| BERT-Base | RTE | [textattack/bert-base-uncased-RTE](https://huggingface.co/textattack/bert-base-uncased-RTE) |
| BERT-Base | STS-B | [textattack/bert-base-uncased-STS-B](https://huggingface.co/textattack/bert-base-uncased-STS-B) |
| LLaMA2-7B | Zero-shot reasoning evaluation | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| SegFormer-B0 | ADE20K semantic segmentation | [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) |

> **Note**  
> For BERT-Base, the repository uses task-specific fine-tuned checkpoints.  
> Please match the checkpoint name to the target GLUE task in your script.

---

## Quantization Setting

All methods are evaluated under the same tiled INT8 framework.  
For every experiment, **input tiles are quantized row-wise, and weight tiles are quantized column-wise**.  
The compared methods differ only in how partial sums are quantized during accumulation.

---

## Experimental Results

To keep the README concise, we report only the final results of the proposed **DPSQ** method below. Full comparisons against all baselines are provided in the paper.

### BERT-Base on GLUE

| Task | Metric | DPSQ |
|---|---:|---:|
| QNLI | Acc. | 91.45 |
| MNLI | Acc. | 84.47 |
| RTE | Acc. | 72.56 |
| STS-B | PCC. | 88.07 |
| MRPC | Acc. | 87.50 |
| CoLA | MCC. | 54.70 |

### LLaMA2-7B Zero-shot Common-Sense Reasoning

| Task | Metric | DPSQ |
|---|---:|---:|
| BoolQ | Acc. | 77.80 |
| PIQA | Acc. | 76.93 |
| HellaSwag | Acc. | 56.89 |
| WinoG. | Acc. | 69.92 |
| Arc-e | Acc. | 75.38 |
| Arc-c | Acc. | 43.43 |
| OBQA | Acc. | 31.60 |

### Vision Models

| Model | Dataset | Metric | DPSQ |
|---|---|---:|---:|
| SegFormer-B0 | ADE20K | mIoU | 36.07 |
| EfficientViT-B1 | ADE20K | mIoU | 42.75 |

### Summary

DPSQ preserves task performance across both language and vision models while applying PSUM quantization under a hardware-friendly tiled INT8 setting. On BERT-Base, DPSQ maintains accuracy close to the floating-point baseline across all GLUE tasks and achieves strong correlation performance on STS-B. On LLaMA2-7B, DPSQ remains robust across a range of zero-shot common-sense reasoning benchmarks. On SegFormer-B0 and EfficientViT-B1, DPSQ maintains semantic segmentation quality with minimal mIoU degradation.


## Installation

### 1. Clone the repository
```bash
git clone --recursive https://github.com/do-one-30/DPSQ.git
cd DPSQ
```
If the repository was cloned without submodules:
```bash
git submodule update --init --recursive
```

### 2. Install dependencies
We recommend Python 3.10.
```bash
pip install -r requirements.txt
```

### 3. Install EfficientViT submodule
For EfficientViT experiments, install the included submodule in editable mode:
```bash
cd efficientvit
pip install -e .
cd ..
```
## Usage
To keep the main README concise, evaluation commands are provided separately as runnable scripts.

Please refer to:

* scripts/README.md

This includes:

* BERT evaluation
* LLaMA evaluation
* SegFormer evaluation
* EfficientViT evaluation
* calibration/evaluation flows for methodB_C

## Reproducibility Notes

### 1. Calibration-based workflow
For methodB_C, evaluation requires two stages:

1. calibration
2. evaluation with the saved scale file

### 2. Hugging Face access
Some models, such as meta-llama/Llama-2-7b-hf, may require prior access approval and authentication through Hugging Face.

### 3. EfficientViT dependency
EfficientViT experiments depend on the Git submodule included in this repository.

### 4. Dataset version note for PIQA
The PIQA dataset may require an older version of the datasets library for stable loading in this codebase. See requirements.txt and the environment notes below.

## Environment Notes
This repository was developed using PyTorch and Hugging Face-based tooling. Since some benchmark datasets and model-loading pipelines can be sensitive to version mismatches, we recommend using the dependency versions listed in requirements.txt.

In particular:

* The datasets package version can affect dataset loading behavior,
* lm-eval should be installed for LLaMA evaluation,
* EfficientViT requires its own local editable installation.

For PIQA, if dataset loading issues occur in your environment, install:

```bash
pip install datasets==3.6.0
```

If you are running experiments that do not require PIQA, a newer datasets version may still work for the other tasks. However, for full reproducibility across all supported tasks, we recommend using the version specified in requirements.txt or preparing a separate environment for PIQA evaluation.

## Output Files

Typical outputs include:

* evaluation result JSON files,
* calibration scale checkpoints (.pt or .pth),
* optional segmentation visualization outputs for EfficientViT.

Examples:

* eval_results_<dataset>.json
* llama_results_<dataset>_<method>.json
* seg_results_<method>.json
* evit_results_<dataset>_<method>.json

## Acknowledgements

This implementation builds on the following open-source tools and libraries:

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* EleutherAI LM Evaluation Harness
* EfficientViT