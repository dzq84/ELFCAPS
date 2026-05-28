"""Audio conditioning adapters for ELF."""

import flax.linen as nn
import jax.numpy as jnp

from modules.layers import DEFAULT_BIAS_INIT, DEFAULT_KERNEL_INIT, RMSNorm


class AudioPrefixAdapter(nn.Module):
    """Project precomputed audio frame features into ELF clean-prefix latents."""

    text_encoder_dim: int
    hidden_dim: int = 1024
    dropout: float = 0.0

    @nn.compact
    def __call__(self, audio_features, audio_feature_mask=None, deterministic=True):
        x = nn.Dense(
            self.hidden_dim, kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT, name="in_proj",
        )(audio_features)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(
            self.text_encoder_dim, kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT, name="out_proj",
        )(x)
        x = RMSNorm(self.text_encoder_dim, name="out_norm")(x)
        if audio_feature_mask is not None:
            x = x * audio_feature_mask[..., None].astype(x.dtype)
        return x

