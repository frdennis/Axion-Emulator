import numpy as np
from typing import Tuple, Optional, List, Type, Sequence, Dict
import jax
import jax.numpy as jnp
import flax.linen as nn

from semu.models.activations_flax import Speculator


class FlaxFullyConnected(nn.Module):
    output_dim: int
    hidden_dims: Sequence[int]
    activation: str

    def setup(self):
        if "custom" in self.activation:
            activation_list = list()
            for i in range(len(self.hidden_dims)):
                activation = Speculator(dim_input=self.hidden_dims[i])
                activation_list.append(activation)
            self.activation_list = activation_list
        else:
            self.activation_func = getattr(nn, self.activation)

    @nn.compact
    def __call__(self, x):
        for i, dims in enumerate(self.hidden_dims):
            x = nn.Dense(dims)(x)
            if "custom" in self.activation:
                x = self.activation_list[i](x)
            else:
                x = self.activation_func(x)
        return nn.Dense(self.output_dim)(x)

    def convert_from_pytorch(self, pt_state: Dict) -> Dict:
        jax_state = dict(pt_state)
        max_hidden = max(
            [int(k.split(".")[1]) for k in pt_state.keys() if "hidden" in k]
        )
        for key, tensor in pt_state.items():
            if "hidden_layers" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                key = key.replace("hidden_layers.", f"Dense_")
                jax_state[key] = tensor.T
            if "output_layer" in key:
                del jax_state[key]

                key = key.replace("output_layer", f"Dense_{max_hidden+1}")
                key = key.replace("weight", "kernel")
                jax_state[key] = tensor.T
            if "activations" in key:
                del jax_state[key]
                _, layer, gamma_or_beta = key.split('.')
                key = f'activation_list_{layer}'
                if key not in jax_state:
                    jax_state[key] = {'gamma': [], 'beta': []}
                jax_state[key][gamma_or_beta] = tensor
        return jax_state
