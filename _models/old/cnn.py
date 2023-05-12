import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple, Optional, List, Type
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from semu.models.moment_loss import moment_loss
from semu.models.activations import Speculator


class CNN(pl.LightningModule):
    """ Fully connected network with forward pass.
    """

    def __init__(
        self,
        n_features: int,
        output_dim: int,
        hidden_dims: List[int] = [250, 250],
        n_fcn: int = 1,
        activation="gelu",
        loss: str = "L1Loss",
        dropout: Optional[float] = None,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        batch_norm: bool = False,
        positive_output: bool = False,
        kernel_size: int = 16,
        stride: int = 1,
        padding: int = 1,
        output_filters: int = 1,
        **kwargs,
    ):

        super(CNN, self).__init__()
        self.save_hyperparameters(
            "n_features",
            "output_dim",
            "hidden_dims",
            "activation",
            "dropout",
            "learning_rate",
            "weight_decay",
            "loss",
            "positive_output",
            "kernel_size",
            "n_fcn",
            "stride",
            "padding",
            "output_filters",
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.n_features = n_features
        self.output_dim = output_dim
        if "custom" not in activation:
            self.activation = getattr(F, activation)
        self.hidden_norms = nn.ModuleList()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        # Stack of hidden layers
        self.n_hidden_layers = len(hidden_dims)
        self.hidden_layers = nn.ModuleList()
        self.activations = []
        self.n_fcn = n_fcn
        output_cnn = hidden_dims[n_fcn - 1]
        for i in range(self.n_hidden_layers):
            if i < n_fcn:
                layer = nn.Linear(
                    in_features=n_features if i == 0 else hidden_dims[i - 1],
                    out_features=hidden_dims[i],
                )
            else:
                layer = nn.Conv1d(
                    in_channels=hidden_dims[i - 1] if i > self.n_fcn else 1,
                    out_channels=hidden_dims[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                output_cnn = (output_cnn - kernel_size) / stride + 1
            self.hidden_layers.append(layer)
            if activation == "custom_speculator":
                self.activations.append(Speculator())
        self.activations = nn.ModuleList(self.activations)
        kernel_size_output = int(
            output_cnn + 2 * padding + stride - stride * self.output_dim
        )
        print('Kernel size output = ', kernel_size_output)
        self.output_layer = nn.Conv1d(
            in_channels=hidden_dims[-1] if n_fcn < len(self.hidden_layers) else 1,
            out_channels=output_filters,
            kernel_size=kernel_size_output,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.hidden_layer_batch_norms = [
                nn.BatchNorm1d(hidden_dims[i]) for i in range(self.n_hidden_layers)
            ]
        self.loss = loss
        if loss == "moment_loss":
            self.loss_fct = moment_loss
        else:
            self.loss_fct = getattr(torch.nn, loss)()
        self.positive_output = positive_output

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--learning_rate", type=float, default=1.0e-3)
        parser.add_argument("--weight_decay", type=float, default=1.0e-4)
        parser.add_argument("--n_hidden", type=int, default=100)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="gelu")
        parser.add_argument("--batch_norm", type=int, default=0)
        return parent_parser

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer.forward(x)
            if i == self.n_fcn - 1:
                x = x[:, None, :]
            if self.activations:
                x = self.activations[i](x)
            else:
                x = self.activation(x)
            if self.batch_norm:
                x = self.hidden_layer_batch_norms[i](x)
            if self.dropout is not None:
                x = self.dropout(x)
        output = self.output_layer.forward(x)
        output = torch.mean(output, dim=1)  # output.squeeze(1)
        if self.positive_output:
            output = torch.exp(output)
        return output

    def custom_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x,)
        return self.loss_fct(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("loss/train", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("loss/test", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("loss/val", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.1,
            min_lr=self.learning_rate * 1.0e-3,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
            "monitor": "loss/val",
        }
