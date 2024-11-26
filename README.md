
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

## Tutorial
There is a tutorial notebook [tutorial.ipynb](tutorial.ipynb) that demonstrates how to train the model with a single slice dataset. For more advanced usage using command line, please refer to the following sections:

### For the dataset with a single slice
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=example.yaml
```

The predicted spatial domain labels will be saved to anndata(.h5ad) files in logs/logger_logs folder. The raw predicted spatial domain labels is in adata.obs["pred_labels"]. The autoencoder refined labels is in adata.obs["mlp_fit"].


The detected spatially variable genes will be saved in adata.uns["svg_dict"].

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20
```

### For the large dataset (multiple GPUs)
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/) with multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py experiment=example_ddp.yaml trainer.devices=2
```

To train model with your own dataset, you can copy the [configs/experiment/example_ddp.yaml](configs/experiment/example_ddp.yaml) to [configs/experiment/your_experiment.yaml](configs/experiment/your_experiment.yaml) file and modify it to your needs. The required data format is h5ad, which can be created by [AnnData](https://anndata.readthedocs.io/en/latest/). The "spatial" key in the obsm attribute of the anndata object (`adata.obsm["spatial"]`) indicates spatial coordinates and is necessary for constructing spatial adjacency graph. The full path to h5ad file is `data_dir/dataset_dir/data_file_name`. You can also specify the requred number of spatial domains with the parameter `num_classes` in your_experiment.yaml as well. The config file has rich comments for explaining the parameters.

```bash
cp configs/experiment/example_ddp.yaml configs/experiment/your_experiment.yaml
python run.py experiment=your_experiment.yaml
```

### For the dataset with a multiple slices
To train with a dataset with multiple slices, you need to first align the dataset with paste2. Refer to [align_multiple_slices_with_paste2.ipynb](align_multiple_slices_with_paste2.ipynb) for preprocessing steps. You can then train with [configs/experiment/example_multi_slices.yaml](configs/experiment/example_multi_slices.yaml). For your own dataset, make sure the obs attribute of the anndata object has the "batch" column (`adata.obs["batch"]`), which indicates the slice index. Set `z_scale` with a meaningful value (refer to config file for details) as `adata.obs["batch"] * z_scale * min_two_units_xy_distance` will be considered as the third coordinate for constructing spatial adjacency graph besides two coordinates in `adata.obsm["spatial"]`.

```bash
python run.py experiment=example_multi_slices.yaml
```

### For reproducing the results in the paper
You can check https://doi.org/10.5281/zenodo.11315102 to download the processed data and reproducible Jupyter notebooks. Please read the README.md inside the zip file for details.
