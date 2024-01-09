
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

To train model with your own dataset, you can copy the [configs/experiment/example.yaml](configs/experiment/example.yaml) to [configs/experiment/your_experiment.yaml](configs/experiment/your_experiment.yaml) file and modify it to your needs. The required data format is h5ad, which can be created by [AnnData](https://anndata.readthedocs.io/en/latest/). The full path to h5ad file is `data_dir/dataset_dir/data_file_name`. You can also specify the requred number of spatial domains with the parameter `num_classes` in your_experiment.yaml as well.

```bash
cp configs/experiment/example.yaml configs/experiment/your_experiment.yaml
python run.py experiment=your_experiment.yaml
```
