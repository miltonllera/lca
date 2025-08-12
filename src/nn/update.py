import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from typing import Callable
from jaxtyping import Float, Array


def stochastic_update(state_shape: tuple[int, ...], update_prob: float, key: jax.Array):
    _, H, W = state_shape
    return jnp.floor(jr.uniform(key, (1, H, W)) + update_prob)


def max_pool_alive(state: Float[Array, "C H W"], alive_index, alive_threshold):
    alive_value = jnp.pad(state[alive_index], pad_width=(1, 1), mode='wrap')[None]
    max_alive = nn.MaxPool2d(kernel_size=3, padding=0)(alive_value)
    return (max_alive  > alive_threshold).astype(jnp.float32)


class GrowingUpdate(eqx.Module):
    update_fn: Callable
    alive_threshold: float
    alive_index: int
    update_prob: float
    max_pool: nn.MaxPool2d

    def __init__(self, update_fn, alive_threshold=0.1, alive_index=3, update_prob=0.5):
        assert 0 < alive_threshold < 1.0
        assert 0 < update_prob <= 1.0

        self.update_fn = update_fn
        self.alive_threshold = alive_threshold
        self.alive_index = alive_index
        self.max_pool = nn.MaxPool2d(kernel_size=3, padding=1)
        self.update_prob = update_prob

    def __call__(
        self,
        state: Float[Array, "C H W"],
        perception: Float[Array, "C H W"],
        key: jax.Array
    ):
        pre_alive = max_pool_alive(state, self.alive_index, self.alive_threshold)
        update_mask = stochastic_update(state.shape, self.update_prob, key)
        state = state + self.update_fn(perception) * update_mask
        alive = max_pool_alive(state, self.alive_index, self.alive_threshold) * pre_alive
        return state * alive

