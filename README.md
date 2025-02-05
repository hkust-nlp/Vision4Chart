


<div align="center">

# On the Perception Bottleneck of LVLMs for Chart Understanding

</div>

Welcome! This repository accompanies the paper **On the Perception Bottleneck of LVLMs for Chart Understanding**.

## What is included in this project?

This repository provides implementations for training and evaluating CLIP and LLaVA models on chart understanding tasks. Specifically, it includes:

- **CLIP Training**: Training scripts for CLIP with and without hard negative captions.
- **CLIP Evaluation**: Code for evaluating CLIP on various chart-related datasets.
- **LLaVA Training**: Training scripts for LLaVA-13B and LLaVA-Phi.
- **LLaVA Evaluation**: Evaluation scripts for LLaVA on multiple chart benchmarks.

## Environment Setup

Detailed instructions for setting up the environment are provided in [`config_env.md`](https://github.com/hkust-nlp/Vision4Chart/blob/main/config_env.md).

## CLIP Training

We utilize the [open_clip](https://github.com/mlfoundations/open_clip) repository for CLIP training. The source code is available in the `open_clip` directory.

Example training script: [`example_scripts/train_openclip.sh`](https://github.com/hkust-nlp/Vision4Chart/blob/main/example_scripts/train_openclip.sh).

For NegCLIP training, we build upon the [neg_clip](https://github.com/vinid/neg_clip) repository, modifying it to support multi-GPU training. The modified code is in the `neg_clip` directory.

Example NegCLIP training script: [`example_scripts/train_negclip.sh`](https://github.com/hkust-nlp/Vision4Chart/blob/main/example_scripts/train_negclip.sh).

## CLIP Evaluation

The evaluation code for CLIP is located in the `eval_clip` directory.

Example evaluation script: [`example_scripts/eval_clip.sh`](https://github.com/hkust-nlp/Vision4Chart/blob/main/example_scripts/eval_clip.sh).

## LLaVA Training

We train two types of LLaVA models:
- **LLaVA-v1.5-13B**: Uses Vicuna-13B as the language model.
- **LLaVA-Phi**: Uses [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) as the language model.

LLaVA-v1.5-13B training is based on the [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file) repository, while LLaVA-Phi training is based on the [LLaVA-pp](https://github.com/mbzuai-oryx/LLaVA-pp) repository. Additionally, we enable unfreezing vision encoder tuning.

Example training full llava script: [`example_scripts/train_full_llava.sh`](https://github.com/hkust-nlp/Vision4Chart/blob/main/example_scripts/train_full_llava.sh).

## LLaVA Evaluation

LLaVA is evaluated on multiple chart-related benchmarks.

For **FigureQA, DVQA, PlotQA, ChartQA, ChartBench, and ChartX**, evaluation scripts are provided in: [`example_scripts/eval_llava.sh`](https://github.com/hkust-nlp/Vision4Chart/blob/main/example_scripts/eval_llava.sh).

For **MathVista**, evaluation scripts are provided in: [`example_scripts/eval_mathvista.sh`](https://github.com/hkust-nlp/Vision4Chart/blob/main/example_scripts/eval_mathvista.sh).

