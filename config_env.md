# How to config different ENVs?

## ENV1: openclip
`openclip` environment is used for CLIP training without hard negatives.
```bash
conda create -n openclip python=3.10
conda activate openclip
cd open_clip
make install
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
make install-training
cd ..
pip install wandb
pip uninstall numpy
pip uninstall numpy
pip install numpy==1.26.4
```

## ENV2: negclip
`negclip` enviroment is used for NegCLIP training

```bash
conda create -n negclip python=3.10
conda activate negclip
cd neg_clip
make install
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
cd ..
pip install braceexpand
pip uninstall numpy  
pip uninstall numpy  
pip install numpy==1.26.4
pip install pandas==1.4.2
pip install webdataset
pip install wandb 
```

##  ENV3：llava
`llava` enviroment is used for LLaVA training, bascially we follow the LLaVA repo.

```bash
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```