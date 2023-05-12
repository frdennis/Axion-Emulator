from typing import List, Optional, Union, Tuple
from pydantic import BaseModel


class DataConfig(BaseModel):
    feature_columns: List[str]
    label_columns: List[str]
    batch_size: Optional[int]
    num_workers: Optional[int]
    shuffle: Optional[bool]
    train_data_path: "str"
    val_data_path: "str"
    test_data_path: "str"
    feature_scaler: "str"
    label_scaler: Optional["str"]



class ModelConfig(BaseModel):
    type: str
    hidden_dims: Optional[List[int]]
    activation: Optional[str]
    loss: Optional[str]
    dropout: Optional[float]
    learning_rate: float
    weight_decay: Optional[float]
    batch_norm: Optional[bool]
    positive_output: bool = False
    kernel_size: Optional[Union[int, Tuple]]
    kernel_size_left: Optional[int]
    kernel_size_right: Optional[int]
    n_fcn: Optional[int]
    stride: Optional[int]
    padding: Optional[int]
    output_filters: Optional[int]
    autoencoder: Optional[bool]
    input_height: Optional[int]
    latent_dim: Optional[int]


class TrainingConfig(BaseModel):
    gpus: Optional[List[int]]
    max_epochs: int
    stochastic_weight_avg: bool
    default_root_dir: str
