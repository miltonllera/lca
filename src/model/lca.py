from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from typing import Literal
from jaxtyping import Float, Array

from src.nn.ca import CellularAutomata
from src.nn.perception import sobel_perception
from src.nn.update import GrowingUpdate
from src.nn.conditioning import PromptConditioning
from src.nn.morphogens import (
    gaussian_field,
    directional_fields,
    sinusoidal_fields,
    mix_fields,
)


class AttentionLCA(eqx.Module):
    state_size: int
    ca: CellularAutomata
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        token_dim: int,
        hidden_size,
        perception_type: Literal['sobel', 'sobel-with-laplace', 'learned'] = 'sobel-with-laplace',
        morphogen_type: Literal['gaussian', 'directional', 'sinusoidal', 'mixed'] = 'mixed',
        update_width = 128,
        update_depth = 1,
        update_prob = 0.5,
        alive_threshold = 0.1,
        alive_index = 3,
        num_dev_steps = (48, 96),
        *,
        key
    ) -> None:
        super().__init__()

        state_size = hidden_size + 4
        conv_key, update_key, cond_key = jr.split(key, 3)

        # Perception function
        if perception_type in ['sobel', 'sobel-with-laplace']:
            perception_fn = partial(sobel_perception, use_laplace='laplace' in perception_type)
        else:
            perception_fn = nn.Conv2d(
                in_channels=state_size,
                out_channels=state_size,
                kernel_size=3,
                padding=1,
                padding_mode='wrap',
                groups=state_size,
                key=conv_key
            )

        # Mophogen init
        if morphogen_type == 'gaussian':
            morphogen_fn = partial(gaussian_field, sigma=1.0)
        elif  morphogen_type == 'directional':
            morphogen_fn = partial(directional_fields, n=2)
        elif morphogen_type == 'sinusoidal':
            morphogen_fn = partial(sinusoidal_fields, channels=4)
        else:
            morphogen_fn = partial(
                mix_fields,
                n=4,
                gaussian_sigma=5.0,
                sin_freq_min=0.5,
                sin_freq_max=1.0,
            )

        morphogen_size = morphogen_fn(8, 8)[0].shape[0]
        morphogen_concat = lambda x: jnp.concat([x, morphogen_fn(*x.shape[1:])[0]])

        # Update function
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state, key=conv_key).shape[0]

        layer_input_size = perception_out_size + morphogen_size
        layers: list[eqx.Module] = [nn.Lambda(morphogen_concat)]

        for _ in range(update_depth):
            update_depth, conv_key = jr.split(update_key)
            layers.extend([
                nn.Conv2d(layer_input_size, update_width, kernel_size=1, key=conv_key),
                nn.Lambda(jax.nn.relu),
            ])
            layer_input_size = update_width

        layers.append(
            nn.Conv2d(layer_input_size, state_size, kernel_size=1, key=update_key)
        )

        layers[-1] = eqx.tree_at(
            where=lambda l: l.weight,
            pytree=layers[-1],
            replace_fn=lambda w: jnp.zeros_like(w)
        )

        update_fn = GrowingUpdate(
            nn.Sequential(layers), # type: ignore
            alive_threshold,
            alive_index,
            update_prob
        )

        # Conditioning function
        conditioning_fn = PromptConditioning(
            state_size,
            token_dim,
            num_heads=1,
            dropout=0.0,
            key=cond_key,
        )

        self.state_size = state_size
        self.ca = CellularAutomata(perception_fn, update_fn, conditioning_fn)
        self.num_dev_steps = num_dev_steps

    def __call__(
        self,
        init_state: Float[Array, "C H W"],
        prompt: Float[Array, "S E"],
        key: jax.Array,
        steps=None
    ):
        if steps is None:
            steps = self.num_dev_steps
        cell_states, dev_path = self.ca(init_state, prompt, steps, key=key)
        return cell_states[:4], dev_path

