import os
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from typing import Callable
from jaxtyping import Array, Float, PyTree

from src.model.lca import AttentionLCA
from src.nn.seeding import init_central_seed
from src.dataset.emojis import EmojiDataset
from src.visualisation.utils import plot_examples
from src.utils.base import filter_put, get_sharding_specs, init_wandb, seed_everything, save_pytree


def main(
    dataset_path: str | Path,
    hidden_size: int,
    update_width: int | None,
    update_depth: int,
    batch_size: int,
    training_iters: int,
    learning_rate: float,
    use_lr_schedule: bool = False,
    save_folder: str | Path = 'data/logs/temp',
    seed: int | None = None,
    *,
    run,
):
    # setup
    rng, jax_key = seed_everything(seed)
    batch_sharding, model_sharding = get_sharding_specs()

    dataset = EmojiDataset(dataset_path, batch_size, rng)

    lca = AttentionLCA(
        token_dim=dataset.token_dim,
        hidden_size=hidden_size,
        update_width=dataset.token_dim * 4 if update_width is None else update_width,
        update_depth=update_depth,
        key=jax_key,
    )

    if model_sharding is not None:
        lca = filter_put(lca, model_sharding)

    def init_fn(b, shape):
        return np.repeat(init_central_seed((lca.state_size, *shape))[None], repeats=b, axis=0)

    if use_lr_schedule:
        warmup_iters = int(training_iters * 0.1)
        lr_or_schedule = optax.warmup_cosine_decay_schedule(
            0.0, learning_rate, warmup_iters, training_iters, end_value=5e-6, exponent=1.25,
        )
        # lr_or_schedule = optax.warmup_constant_schedule(0.0, learning_rate, warmup_iters)
    else:
        lr_or_schedule = learning_rate

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        # optax.clip_by_block_rms(1.0),
        optax.adamw(lr_or_schedule),
    )
    opt_state = optim.init(eqx.filter(lca, eqx.is_array))

    def compute_loss(
        model: Callable,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array
    ):
        inputs, targets = batch
        B, _, H, W = targets.shape

        batch_key = jr.split(key, B)
        init_states = init_fn(B, (H, W))
        preds, _ = jax.vmap(model)(init_states, inputs, batch_key)

        return jnp.sum(optax.l2_loss(preds, targets)) / len(targets)

    @eqx.filter_jit(donate='all')
    def train_step(
        model: PyTree,
        opt_state: PyTree,
        batch: tuple[Float[Array, "NCHW"], Float[Array, "NCHW"]],
        key: jax.Array,
    ):
        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, batch, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for i in (pbar := tqdm(range(training_iters))):
        jax_key, step_key = jr.split(jax_key, 2)
        batch = dataset.sample_batch()
        if batch_sharding is not None:
            batch = filter_put(batch, batch_sharding)
        lca, opt_state, train_loss = train_step(lca, opt_state, batch, step_key)  #type: ignore
        pbar.set_postfix_str(f"iter: {i}; loss: {np.asarray(train_loss)}")
        run.log({"mse": float(train_loss)})

    batch_key = jr.split(jax_key, 8)
    prompts, targets = dataset[list(range(8))]
    H, W = targets.shape[-2:]
    preds, _ = jax.vmap(lca)(init_fn(8, (H, W)), prompts, batch_key)  # type: ignore

    save_folder = Path(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    plot_examples(preds, w=4, format='NCHW').savefig(save_folder / "examples.png")
    save_pytree(lca, save_folder / "checkpoint.eqx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LCA model.")

    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the dataset file."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Size of the hidden state in the LCA model."
    )
    parser.add_argument(
        "--update_width",
        type=int,
        default=None,
        help="Width of the update MLP in the LCA model. " \
            "If None, it will be set to 4 times the token dimension."
    )
    parser.add_argument(
        "--update_depth",
        type=int,
        default=2,
        help="Depth of the update layers in the LCA model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training."
    )
    parser.add_argument(
        "--training_iters",
        type=int,
        default=1000,
        help="Number of training iterations."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--use_lr_schedule",
        action='store_true',
        help="Whether to use a learning rate schedule."
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default='data/logs/temp',
        help="Folder to save the training logs and model checkpoints."
    )

    args = vars(parser.parse_args())
    with init_wandb(**args) as run:
        main(**args, run=run)

