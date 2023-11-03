# import faulthandler
# faulthandler.enable()

import hydra
# import rootutils
import pyrootutils as rootutils
from omegaconf import DictConfig
import warnings
warnings.filterwarnings("ignore", message="Variable names are not unique. ")
warnings.filterwarnings("ignore", message="To make them unique, call `.var_names_make_unique`.")
warnings.filterwarnings("ignore", message=".*storing*as categorical.*")
warnings.filterwarnings("ignore", message=".*PyTorch skipping the first value of the learning rate schedule.*")
warnings.filterwarnings("ignore", message=".*remember to set when you upload.*")
warnings.filterwarnings("ignore", message=".*Unknown error exporting current conda environment")
warnings.filterwarnings("ignore", message=".*Unknown error retrieving Conda package as an explicit file")
warnings.filterwarnings("ignore", message=".*Unknown error retrieving Conda information")
warnings.filterwarnings("ignore", message=".*could be inaccurate if each worker is not configured independently to avoid having duplicate data.*")
warnings.filterwarnings("ignore", message=".* is smaller than the logging interval Trainer.*")
warnings.filterwarnings("ignore", message=".*does not have many workers which may be a bottleneck.*")
warnings.filterwarnings("ignore", message=".*srun.*")
warnings.filterwarnings("ignore", message=".*Created a temporary directory.*")
warnings.filterwarnings("ignore", message=".*_remote_module_non_scriptable.*")
warnings.filterwarnings("ignore", message=".*on epoch level in distributed setting to accumulate the metric across devices.*")

# import comet_ml

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)

# @hydra.main(config_path="configs/", config_name="config.yaml")
# @hydra.main(config_path=root / "configs", config_name="config.yaml")
# @hydra.main(version_base="1.2", config_path=root / "configs", config_name="config.yaml")
@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
