#!/usr/bin/env python
"""Generate AudioCaps captions from an ELF audio-prefix checkpoint."""

import argparse
import json
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config import load_config_from_yaml, load_sampling_configs
from modules.audio_adapter import AudioPrefixAdapter
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_checkpoint
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset
from utils.generation_utils import (
    _generate_samples_single_batch,
    _dlm_decode_batch,
    _shard_noise,
    _shard_timesteps,
    mask_after_eos,
    shift_left,
)
from utils.caption_postprocess import clean_caption_artifacts
from utils.train_utils import TrainState, get_optimizer, create_learning_rate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an AudioCaps ELF checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_sampling_steps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--max_decode_tokens",
        type=int,
        default=None,
        help="Optional cap on generated text tokens after removing the audio prefix.",
    )
    parser.add_argument(
        "--min_decode_tokens",
        type=int,
        default=0,
        help="Suppress EOS before this many generated text tokens.",
    )
    parser.add_argument(
        "--decode_latent_scale",
        type=float,
        default=1.0,
        help="Scale decoded text latents after the audio prefix; 1.0 preserves checkpoint behavior.",
    )
    parser.add_argument(
        "--sample_text_rms_scale",
        type=float,
        default=1.0,
        help="Scale text latents after each sampling step; diagnostic only, 1.0 preserves checkpoint behavior.",
    )
    parser.add_argument(
        "--time_schedule",
        choices=("logit_normal", "uniform"),
        default=None,
        help="Override the sampling timestep schedule from the sampling config.",
    )
    parser.add_argument(
        "--use_raw_params",
        action="store_true",
        help="Use non-EMA model/audio parameters from the checkpoint.",
    )
    parser.add_argument(
        "--raw_param_mix",
        type=float,
        default=0.0,
        help="Interpolate EMA and raw checkpoint params: 0.0=EMA, 1.0=raw. Overrides --use_raw_params when >0.",
    )
    parser.add_argument(
        "--interp_checkpoint",
        default=None,
        help="Optional second checkpoint for cross-checkpoint parameter interpolation.",
    )
    parser.add_argument(
        "--interp_alpha",
        type=float,
        default=0.0,
        help="Cross-checkpoint interpolation weight: 0.0=--checkpoint, 1.0=--interp_checkpoint.",
    )
    parser.add_argument(
        "--interp_use_raw_params",
        action="store_true",
        help="Use raw params from --interp_checkpoint; otherwise use its EMA params.",
    )
    parser.add_argument("--clean_artifacts", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.eval_data_path is not None:
        config.eval_data_path = args.eval_data_path
    config.eval_data_limit = max(config.eval_data_limit or 0, args.num_samples)
    sampling_config = load_sampling_configs(config.sampling_configs_path)[0]
    sampling_config.num_sampling_steps = [args.num_sampling_steps]
    sampling_config.cfgs = [args.cfg_scale]
    sampling_config.self_cond_cfg_scales = [args.self_cond_cfg_scale]
    if args.time_schedule is not None:
        sampling_config.time_schedule = args.time_schedule

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    _, eval_dataset = load_dataset(config)
    if eval_dataset is None:
        raise ValueError("Audio evaluation requires eval_data_path.")

    encoder_config, _, _ = get_encoder(config.encoder_model_name, jnp.float32)
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=len(tokenizer),
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    audio_adapter = AudioPrefixAdapter(
        text_encoder_dim=encoder_config.d_model,
        hidden_dim=config.audio_adapter_hidden_dim,
        dropout=config.audio_adapter_dropout,
    )

    rng = jax.random.PRNGKey(config.seed)
    input_dim = 2 * encoder_config.d_model if config.self_cond_prob > 0 else encoder_config.d_model
    dummy_x = jnp.ones((1, config.max_length, input_dim))
    dummy_t = jnp.ones((1,))
    dummy_self_cond_cfg_scale = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
    elf_vars = model.init(
        rng, dummy_x, dummy_t, deterministic=True,
        self_cond_cfg_scale=dummy_self_cond_cfg_scale,
    )
    dummy_audio = jnp.ones((1, config.audio_num_prefix_tokens, config.audio_feature_dim))
    dummy_audio_mask = jnp.ones((1, config.audio_num_prefix_tokens))
    audio_vars = audio_adapter.init(rng, dummy_audio, dummy_audio_mask, deterministic=True)

    optimizer = get_optimizer(
        config,
        create_learning_rate_fn(1, 0, config.lr or 1e-6),
        grad_accum_steps=1,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=elf_vars["params"],
        tx=optimizer,
        dropout_rng=rng,
        ema_params1=elf_vars["params"],
        audio_adapter_params=audio_vars["params"],
        audio_adapter_opt_state=optimizer.init(audio_vars["params"]),
        ema_audio_adapter_params1=audio_vars["params"],
    )
    state, _ = load_checkpoint(args.checkpoint, state)
    interp_state = None
    if args.interp_checkpoint is not None:
        if not 0.0 <= args.interp_alpha <= 1.0:
            raise ValueError("--interp_alpha must be in [0.0, 1.0]")
        interp_state, _ = load_checkpoint(args.interp_checkpoint, state)

    num_devices = jax.local_device_count()
    per_device = max(1, args.batch_size // num_devices)
    effective_batch_size = per_device * num_devices
    dataloader = get_dataloader(
        eval_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=pad_token_id,
        distributed=False,
        audio_conditioning=True,
        audio_feature_key=config.audio_feature_key,
        audio_num_prefix_tokens=config.audio_num_prefix_tokens,
    )

    if args.raw_param_mix > 0:
        alpha = float(args.raw_param_mix)
        if alpha > 1.0:
            raise ValueError("--raw_param_mix must be <= 1.0")
        model_source_params = jax.tree_util.tree_map(
            lambda ema, raw: ema * (1.0 - alpha) + raw * alpha,
            state.ema_params1,
            state.params,
        )
        audio_source_params = jax.tree_util.tree_map(
            lambda ema, raw: ema * (1.0 - alpha) + raw * alpha,
            state.ema_audio_adapter_params1,
            state.audio_adapter_params,
        )
    else:
        model_source_params = state.params if args.use_raw_params else state.ema_params1
        audio_source_params = (
            state.audio_adapter_params if args.use_raw_params else state.ema_audio_adapter_params1
        )
    if interp_state is not None:
        alpha = float(args.interp_alpha)
        interp_model_params = (
            interp_state.params if args.interp_use_raw_params else interp_state.ema_params1
        )
        interp_audio_params = (
            interp_state.audio_adapter_params
            if args.interp_use_raw_params else interp_state.ema_audio_adapter_params1
        )
        model_source_params = jax.tree_util.tree_map(
            lambda a, b: a * (1.0 - alpha) + b * alpha,
            model_source_params,
            interp_model_params,
        )
        audio_source_params = jax.tree_util.tree_map(
            lambda a, b: a * (1.0 - alpha) + b * alpha,
            audio_source_params,
            interp_audio_params,
        )
    model_params = jax_utils.replicate(model_source_params)
    audio_adapter_params = jax_utils.replicate(audio_source_params)
    p_generate = jax.pmap(
        partial(
            _generate_samples_single_batch,
            model_apply_fn=model.apply,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=args.cfg_scale,
            self_cond_cfg_scale=args.self_cond_cfg_scale,
            sample_text_rms_scale=args.sample_text_rms_scale,
        )
    )
    p_decode = jax.pmap(
        partial(
            _dlm_decode_batch,
            model_apply_fn=model.apply,
            config=config,
            self_cond_cfg_scale=args.self_cond_cfg_scale,
            min_decode_tokens=args.min_decode_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            decode_latent_scale=args.decode_latent_scale,
        )
    )
    p_audio_adapter = jax.pmap(
        partial(audio_adapter.apply, deterministic=True),
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(dataloader):
            if written >= args.num_samples:
                break
            batch_size_current = batch["input_ids"].shape[0]
            if batch_size_current % num_devices != 0:
                pad_size = num_devices - (batch_size_current % num_devices)
                for key in batch:
                    if isinstance(batch[key], np.ndarray):
                        pad_arr = np.zeros((pad_size,) + batch[key].shape[1:], dtype=batch[key].dtype)
                        batch[key] = np.concatenate([batch[key], pad_arr], axis=0)
                    elif isinstance(batch[key], list):
                        batch[key] = batch[key] + [""] * pad_size

            actual_batch_size = batch["input_ids"].shape[0]
            current_per_device = actual_batch_size // num_devices
            audio_features = jnp.array(batch["audio_features"]).reshape(
                num_devices, current_per_device, config.audio_num_prefix_tokens, config.audio_feature_dim
            )
            audio_mask = jnp.array(batch["audio_feature_mask"]).reshape(
                num_devices, current_per_device, config.audio_num_prefix_tokens
            )
            cond_audio = p_audio_adapter(
                {"params": audio_adapter_params},
                audio_features,
                audio_mask,
            )
            target_len = config.max_length - config.audio_num_prefix_tokens
            zeros_tail = jnp.zeros((num_devices, current_per_device, target_len, encoder_config.d_model))
            cond_seq = jnp.concatenate([cond_audio, zeros_tail], axis=2)
            cond_mask = jnp.concatenate(
                [audio_mask, jnp.zeros((num_devices, current_per_device, target_len))],
                axis=2,
            )

            batch_rng = jax.random.fold_in(rng, batch_idx)
            noise_rng, t_rng = jax.random.split(batch_rng)
            device_rngs = jax.random.split(noise_rng, num_devices)
            t_steps = _shard_timesteps(
                t_rng, num_devices, args.num_sampling_steps,
                sampling_config.time_schedule, config,
            )
            z = _shard_noise(
                device_rngs, num_devices, current_per_device,
                config.max_length, encoder_config.d_model, config.denoiser_noise_scale,
            )
            latent = p_generate(
                model_params=model_params,
                rng=device_rngs,
                z=z,
                t_steps=t_steps,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            predicted_ids = p_decode(
                z=latent,
                model_params=model_params,
                t_final_val=t_steps[:, -1],
            ).reshape(-1, config.max_length)
            predicted_ids = shift_left(
                predicted_ids,
                jnp.full((predicted_ids.shape[0],), config.audio_num_prefix_tokens),
                pad_token_id,
            )[:, :target_len]
            raw_predicted_ids = predicted_ids
            if args.max_decode_tokens is not None:
                decode_cap = max(1, min(args.max_decode_tokens, target_len))
                keep_positions = jnp.arange(target_len)[None, :] < decode_cap
                predicted_ids = jnp.where(keep_positions, predicted_ids, pad_token_id)
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id, pad_token_id)

            for i in range(min(batch_size_current, actual_batch_size)):
                if written >= args.num_samples:
                    break
                raw_ids = np.array(raw_predicted_ids[i])
                final_ids = np.array(predicted_ids[i])
                eos_positions = np.where(raw_ids == eos_token_id)[0]
                first_eos_position = int(eos_positions[0]) if len(eos_positions) else None
                prediction = tokenizer.decode(final_ids, skip_special_tokens=True)
                if args.clean_artifacts:
                    prediction = clean_caption_artifacts(prediction)
                f.write(json.dumps({
                    "input": batch["input"][i],
                    "target": batch["target"][i],
                    "prediction": prediction,
                    "raw_nonpad_tokens": int(np.sum(raw_ids != pad_token_id)),
                    "final_nonpad_tokens": int(np.sum(final_ids != pad_token_id)),
                    "first_eos_position": first_eos_position,
                    "has_eos": first_eos_position is not None,
                }, ensure_ascii=False) + "\n")
                written += 1


if __name__ == "__main__":
    main()
