import json
import typing
from typing import Tuple, Literal
import hashlib

from pydantic import BaseModel


class AEConfig(BaseModel):
    name: str = 'ae'
    embedding_dim: int = 128
    compress_dims: Tuple[int, ...] = (128, 128)
    decompress_dims: Tuple[int, ...] = (128, 128)
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
    mode: Literal['AE', 'VAE'] = 'AE'
    model: Literal['VAE', 'FVAE'] = 'AE'
    use_transformer: bool = True
    x_data_shape: int = None
    t_data_shape: int = None
    y_data_shape: int = None
    principal_elem: Literal['x', 't', 'y'] = 'x'
    patience: int = 40
    shuffle_data: bool = True,
    data_test_rate: float = 0.33
    data_val_rate: float = 0.05

    datamodule_elem_index: dict = None

    def datamodule_principal_elem_index(self):
        return self.datamodule_elem_index[self.principal_elem]

    @property
    def input_shape(self):
        if self.principal_elem == 'x':
            return self.x_data_shape
        elif self.principal_elem == 't':
            return self.t_data_shape
        elif self.principal_elem == 'y':
            return self.y_data_shape

    def hash(self):
        elems = [getattr(self, k) for k, v in self.schema().get("properties").items() if 'type' in v]
        elems = "".join([json.dumps(e) if isinstance(e, dict) else str(e) for e in elems])
        return hashlib.md5(elems.encode("utf-8")).hexdigest()


