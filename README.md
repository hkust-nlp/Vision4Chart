

<div align="center">

# On the Perception Bottleneck of LVLMs for Chart Understanding


</div>

Welcome! Here is the project for the paper **On the Perception Bottleneck of LVLMs for Chart Understanding**.

## What are included in this project?

- **CLIP Training**: Both with and without hard negative captions are included.

- **CLIP Evaluation**: Code for conducting evaluation on Chart datasets

- **LLaVA Training**: Training LLaVA-13B and LLaVA-Phi

- **LLaVA Evaluation**: Evaluation of LLaVA on Chart benchmarks

## CLIP Training

We adopt the [open_clip](https://github.com/mlfoundations/open_clip) repo for our CLIP training.


For NegCLIP training, build on the [neg_clip](https://github.com/vinid/neg_clip), we edit it to support multi-gpu training.


## CLIP Evaluation


## LLaVA Training

We use two type LLaVA models, LLaVA-v1.5-13B, where the LLM part use Vicuna-13B, and LLaVA-Phi, where the LLM part use [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

The LLaVA-v1.5-13B training adopt the from the [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file) repo. The LLaVA-Phi training use the repo [LLaVA-pp](https://github.com/mbzuai-oryx/LLaVA-pp).

## LLaVA Evaluation
