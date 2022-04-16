from typing import Tuple, Literal

from pydantic import BaseModel
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm
import pathlib
import wandb
from pytorch_lightning.loggers import WandbLogger
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fair_vae.datamodule import VAEData


class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


pl_bar = LitProgressBar()

artifacts_path = pathlib.Path.cwd().joinpath('artifacts')

working_dir = "D:/vaun_/PycharmProjects/leoleo"


def set_wandb(config, project, job_type):
    wandb.init(project=project, job_type=job_type)
    wandb_logger = WandbLogger()
    wandb.config = config
    return wandb_logger


class MetricTracker(Callback):

    def __init__(self):
        self.collection = {}

    def safe_append(self, item, elem):
        try:
            self.collection[item].append(float(elem))  # track them
        except:
            self.collection[item] = [float(elem)]

    def on_validation_batch_end(self, trainer, lightning_module, outputs, batch, batch_idx, dataloader_idx):
        for k, v in outputs.items():
            self.safe_append(k, v)

        for k, v in trainer.logged_metrics.items():
            self.safe_append(k, v)

    def on_validation_epoch_end(self, trainer, pl_module):
        for k, v in trainer.logged_metrics.items():
            self.safe_append(k, v)

    def plot(self, item):
        df = pd.DataFrame({"step": np.arange(0, len(self.collection[item])),
                           item: np.array(self.collection[item])})
        sns.lineplot(data=df, x='step', y=item)
        plt.show()


class AEConfig(BaseModel):
    name: str = 'ae'
    embedding_dim: int = 128
    compress_dims: Tuple[int] = (128, 128)
    decompress_dims: Tuple[int] = (128, 128)
    num_resamples: int = 8
    device: Literal['cpu', 'gpu'] = 'cpu'
    lr: float = 1e-4
    batch_size: int = 500
    epochs: int = 100
    no_progress_bar: bool = False
    steps_log_loss: int = 10
    steps_log_norm_params: int = 10
    weight_decay: float = 1e-5
    loss_factor: int = 2
    l2scale: float = 1e-5
    mode: Literal['AE', 'VAE'] = 'AE'
    use_transformer: bool = True
    x_data: VAEData
    t_data: VAEData = None
    y_data: VAEData = None
    patience: int = 40
