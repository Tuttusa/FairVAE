import typing
from typing import Tuple, Literal

from pydantic import BaseModel


class AEConfig(BaseModel):
    name: str = 'ae'
    embedding_dim: int = 128
    compress_dims: Tuple[int] = (128, 128)
    decompress_dims: Tuple[int] = (128, 128)
    num_resamples: int = 8
    device: Literal['cpu', 'gpu'] = 'cpu'
    lr: float = 1e-4
    batch_size: int = 500
    epochs: int = 1
    no_progress_bar: bool = False
    steps_log_loss: int = 10
    steps_log_norm_params: int = 10
    weight_decay: float = 1e-5
    loss_factor: int = 2
    l2scale: float = 1e-5
    fair: bool = False
    mode: Literal['AE', 'VAE'] = 'AE'
    use_transformer: bool = True
    x_data_shape: int = None
    t_data_shape: int = None
    y_data_shape: int = None
    patience: int = 40
    shuffle_data: bool = True,
    data_test_rate: float = 0.33
    data_val_rate: float = 0.05

    def __hash__(self):
        return hash(tuple([getattr(self, k) for k, v in self.schema().get("properties").items() if 'type' in v]))