"""Lightweight latent repair modules for audio-conditioned ELF samples."""

import flax.linen as nn
import jax.numpy as jnp

from modules.layers import DEFAULT_BIAS_INIT, DEFAULT_KERNEL_INIT, RMSNorm


class LatentRepairAdapter(nn.Module):
    """Residual MLP that repairs sampled text latents before DLM decoding."""

    text_encoder_dim: int
    hidden_dim: int = 1024
    dropout: float = 0.0
    residual_scale: float = 0.25
    use_gate: bool = False
    gate_bias: float = -2.0

    @nn.compact
    def __call__(
        self,
        sampled_text_latent,
        audio_prefix_latent,
        audio_mask=None,
        deterministic=True,
        return_gate=False,
    ):
        if audio_mask is None:
            audio_context = jnp.mean(audio_prefix_latent, axis=1)
        else:
            weights = audio_mask[..., None].astype(audio_prefix_latent.dtype)
            audio_context = jnp.sum(audio_prefix_latent * weights, axis=1)
            audio_context = audio_context / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        audio_context = nn.Dense(
            self.text_encoder_dim,
            kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT,
            name="audio_context_proj",
        )(audio_context)
        audio_context = audio_context[:, None, :]
        x = jnp.concatenate(
            [sampled_text_latent, jnp.broadcast_to(audio_context, sampled_text_latent.shape)],
            axis=-1,
        )
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT,
            name="in_proj",
        )(x)
        h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        residual = nn.Dense(
            self.text_encoder_dim,
            kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT,
            name="out_proj",
        )(h)
        residual = RMSNorm(self.text_encoder_dim, name="out_norm")(residual)
        residual_scale = jnp.asarray(self.residual_scale, dtype=sampled_text_latent.dtype)
        if self.use_gate:
            gate_logits = nn.Dense(
                1,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(self.gate_bias),
                name="gate_proj",
            )(h)
            gate = nn.sigmoid(gate_logits).astype(sampled_text_latent.dtype)
        else:
            gate = jnp.ones(sampled_text_latent.shape[:-1] + (1,), dtype=sampled_text_latent.dtype)
        repaired = sampled_text_latent + residual_scale * gate * residual
        if return_gate:
            return repaired, gate
        return repaired
