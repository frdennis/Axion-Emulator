from typing import Any, Callable, Optional, Tuple
import jax
from flax import linen as nn
from flax.linen.initializers import lecun_normal
import jax.numpy as jnp


default_init = jax.nn.initializers.normal()

def speculator_activation(x, gamma, beta):
    """
    https://arxiv.org/pdf/1911.11778.pdf (Eq 8)
    """
    return (gamma + nn.sigmoid(beta * x) * (1.0 - gamma)) * x



class Speculator(nn.Module):
    dim_input: int
    gamma_init: Callable[[Any, Tuple[int,...], Any], Any] = default_init
    beta_init: Callable[[Any, Tuple[int,...], Any], Any] = default_init
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,x):
        gamma = self.param('gamma',
                        self.gamma_init,
                        (self.dim_input,),
                        self.param_dtype)
        gamma = jnp.asarray(gamma)
        beta = self.param('beta',
                        self.beta_init,
                        (self.dim_input,),
                        self.param_dtype)
        beta = jnp.asarray(beta)
        return speculator_activation(x, gamma=gamma, beta=beta)

