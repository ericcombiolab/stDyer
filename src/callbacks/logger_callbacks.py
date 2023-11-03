from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities import rank_zero_only

def get_logger(trainer: Trainer):
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, CometLogger):
        return trainer.logger

    raise Exception(
        "The logger is not CometLogger. "
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100, log_graph=False):
        self.wandb_log = log
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_logger(trainer=trainer)
        # logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=self.log_graph) # buggy self.log is used by other classes
        logger.watch(model=trainer.model, log=self.wandb_log, log_freq=self.log_freq, log_graph=self.log_graph)

class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_logger(trainer=trainer)
        experiment = logger.experiment
        if isinstance(logger, CometLogger):
            experiment.log_code(folder=self.code_dir)
