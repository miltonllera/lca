import os
import datetime
import yaml
from pathlib import Path
from typing import Any
from jaxtyping import PyTree

import numpy as np
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.random as jr
import equinox as eqx
# import wandb


def seed_everything(seed: int | None):
    if seed is None:
        seed = int(datetime.datetime.now().timestamp())

    rng = np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    jax_seed = rng.choice(2 ** 32 - 1)
    np.random.seed(rng.choice(2 ** 32 - 1))

    return rng, jr.key(jax_seed)


#--------------------------------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


#--------------------------------------------- I/O ------------------------------------------------

def save_pytree(model: PyTree, save_file: str | Path):
    save_file = Path(save_file)
    os.makedirs(save_file.parent, exist_ok=True)
    eqx.tree_serialise_leaves(save_file, model)


def load_pytree(checkpoint_file: str | Path, template: PyTree):
    checkpoint_file = Path(checkpoint_file)
    assert checkpoint_file.is_file()
    return eqx.tree_deserialise_leaves(checkpoint_file.with_suffix(".eqx"), template)


#--------------------------------------- Parallelization -----------------------------------------

def get_sharding_specs():
    n_devices = len(jax.devices(backend='gpu'))
    if n_devices > 1:
        mesh = jax.make_mesh((n_devices,), ('batch'))
        sharding = NamedSharding(mesh, P('batch'))
        replicated_sharding = NamedSharding(mesh, P())
    else:
        sharding = None
        replicated_sharding = None

    return sharding, replicated_sharding


def filter_put(model: PyTree, sharding):
    params, static = eqx.partition(model, eqx.is_array)
    params = jax.device_put(params, sharding)
    return eqx.combine(params, static)


#-------------------------------------------- WandB ----------------------------------------------

# def init_wandb(**kwargs):
#     return wandb.init(
#         entity="mlle",
#         project="morphogen-dev",
#         name=f"{kwargs['dataset']}_" \
#              f"{kwargs['model']}_" \
#              f"{kwargs['morphogen_type']}_" \
#              f"{kwargs['perception_type']}",
#         config=kwargs
#     )

