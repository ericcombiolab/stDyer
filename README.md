
---

<div align="center">

# stDyer

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

## Description

stDyer is a spatial domain cluster method for sptailly resolved transcriptomic data.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ericcombiolab/stDyer.git
cd stDyer

# create conda environment
conda env create -f stdyer.yml
conda activate stdyer
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=example.yaml
```

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20
```

Train model with your own dataset

```bash
python run.py experiment=your_experiment.yaml
```
