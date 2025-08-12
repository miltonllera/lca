import functools as ft
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Bool, Float
from equinox.nn._attention import dot_product_attention_weights


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Bool[Array, "q_seq kv_seq"] | None = None,
    dropout: eqx.nn.Dropout | None = None,
    *,
    key: jax.Array | None = None,
    inference: bool | None = None,
) -> tuple[Float[Array, "q_seq v_size"], Float[Array, "q_seq kv_seq"]]:
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn, weights


class MultiheadAttention(eqx.nn.MultiheadAttention):
    """
    A partial re-write of the MultiheadAttention module from Equinox.

    The original implementation does not return the attention weights as part of the module
    outputs. However, we wish to use these weights to understand what the calling module is
    attending to. The rewrite is thus very minimal, just adding an extra output to
    'dot_product_attention' and handling these appropriately in the main function call.
    """
    @jax.named_scope("eqx.nn.MultiheadAttention")
    def __call__(  # type: ignore
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Bool[Array, "q_seq kv_seq"] | Bool[Array, "num_heads q_seq kv_seq"] | None = None,
        *,
        key: jax.Array | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, "q_seq v_size"], Float[Array, "q_seq kv_seq"]]:
        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape

        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        attn_fn = partial(
            dot_product_attention, dropout=self.dropout, inference=inference
        )

        keys = None if key is None else jax.random.split(key, query_heads.shape[1])
        if mask is not None and mask.ndim == 3:
            # Batch `mask` and `keys` down their 0-th dimension.
            attn, weights = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, mask=mask, key=keys
            )
        else:
            # Batch `keys` down its 0-th dimension.
            attn, weights = jax.vmap(ft.partial(attn_fn, mask=mask), in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, key=keys
            )
        attn = attn.reshape(query_seq_length, -1)
        weights = weights.reshape(query_seq_length, -1)  # check that this is correct

        return jax.vmap(self.output_proj)(attn), weights


class PromptConditioning(eqx.Module):
    mh_attn: MultiheadAttention

    def __init__(
        self,
        state_size: int,
        prompt_emb_size: int,
        num_heads: int=1,
        dropout: float=0.0,
        *,
        key
    ):
        self.mh_attn = MultiheadAttention(
            num_heads=num_heads,
            query_size=state_size,
            key_size=prompt_emb_size,
            value_size=prompt_emb_size,
            output_size=state_size,
            qk_size=prompt_emb_size,
            dropout_p=0.0,  # type: ignore
            key=key, # type: ignore
        )

    def __call__(self, inputs, conditioning, key):
        outputs, _ = self.mh_attn(inputs[None], conditioning, conditioning, key=key)
        return outputs[0]  # since we only have one element in the query sequence (the cell)

