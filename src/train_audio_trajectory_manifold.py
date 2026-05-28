#!/usr/bin/env python
"""Trajectory-level teacher-manifold alignment for AudioCaps ELF checkpoints.

The base failure is that audio-conditioned sampled text latents have much lower
RMS than teacher caption latents, while the decoder can decode teacher latents
well. This diagnostic unrolls the actual sampler during training and aligns the
final sampled text latent to the frozen T5 teacher latent with MSE, cosine, and
RMS losses.
"""

import argparse
import json
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config import load_config_from_yaml, load_sampling_configs
from modules.audio_adapter import AudioPrefixAdapter
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_checkpoint, load_encoder_checkpoint, save_checkpoint
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset
from utils.encoder_utils import encode_text
from utils.sampling_utils import restore_cond, _ode_step, get_sampling_steps
from utils.train_utils import TrainState, create_learning_rate_fn, get_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF with trajectory manifold alignment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_sampling_steps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=1.0)
    parser.add_argument("--time_schedule", choices=("logit_normal", "uniform"), default=None)
    parser.add_argument("--mse_weight", type=float, default=0.25)
    parser.add_argument("--cos_weight", type=float, default=0.5)
    parser.add_argument("--rms_weight", type=float, default=1.0)
    parser.add_argument("--mean_weight", type=float, default=0.05)
    parser.add_argument(
        "--step_xpred_mse_weight",
        type=float,
        default=0.0,
        help="Auxiliary MSE from intermediate ODE clean predictions to teacher text latents.",
    )
    parser.add_argument(
        "--step_xpred_cos_weight",
        type=float,
        default=0.0,
        help="Auxiliary cosine loss from intermediate ODE clean predictions to teacher text latents.",
    )
    parser.add_argument(
        "--step_xpred_rms_weight",
        type=float,
        default=0.0,
        help="Auxiliary RMS-ratio loss from intermediate ODE clean predictions to teacher text latents.",
    )
    parser.add_argument(
        "--sampled_decoder_ce_weight",
        type=float,
        default=0.0,
        help="CE on teacher tokens from the final sampled trajectory latent.",
    )
    parser.add_argument(
        "--sampled_decoder_stop_gradient",
        action="store_true",
        help="Stop gradients from sampled-decoder CE into the trajectory latent.",
    )
    parser.add_argument(
        "--eos_loss_weight",
        type=float,
        default=None,
        help="Optional EOS CE weight override for sampled-decoder CE.",
    )
    parser.add_argument(
        "--artifact_unlikelihood_weight",
        type=float,
        default=0.0,
        help="Unlikelihood penalty for decoded artifact continuations such as 'and a', 'ing a', 'aa', and 'ss'.",
    )
    parser.add_argument("--artifact_unlikelihood_margin", type=float, default=1e-4)
    parser.add_argument(
        "--artifact_expected_weight",
        type=float,
        default=0.0,
        help="Differentiable expected bad n-gram probability penalty for sampled decoder logits.",
    )
    parser.add_argument(
        "--artifact_expected_and_a_weight",
        type=float,
        default=1.0,
        help="Multiplier for the expected 'and a' artifact term.",
    )
    parser.add_argument(
        "--artifact_expected_aa_weight",
        type=float,
        default=1.0,
        help="Multiplier for the expected 'aa' artifact term.",
    )
    parser.add_argument(
        "--artifact_expected_ss_weight",
        type=float,
        default=1.0,
        help="Multiplier for the expected 'ss' artifact term.",
    )
    parser.add_argument(
        "--artifact_expected_ing_a_weight",
        type=float,
        default=1.0,
        help="Multiplier for the expected 'ing a' artifact term.",
    )
    parser.add_argument(
        "--artifact_expected_suffix_boost",
        type=float,
        default=0.0,
        help="Extra multiplier for artifact terms ending within the final suffix window.",
    )
    parser.add_argument(
        "--artifact_expected_suffix_window",
        type=int,
        default=2,
        help="Number of trailing token slots treated as suffix-artifact sensitive.",
    )
    parser.add_argument("--freeze_audio_adapter", action="store_true")
    parser.add_argument("--use_ema_init", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=0)
    return parser.parse_args()


def _build_state(config, encoder_config, tokenizer, rng, lr, total_steps, warmup_steps):
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
    input_dim = 2 * encoder_config.d_model if config.self_cond_prob > 0 else encoder_config.d_model
    dummy_self_cond_cfg_scale = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
    elf_vars = model.init(
        rng,
        jnp.ones((1, config.max_length, input_dim)),
        jnp.ones((1,)),
        deterministic=True,
        self_cond_cfg_scale=dummy_self_cond_cfg_scale,
    )
    audio_vars = audio_adapter.init(
        rng,
        jnp.ones((1, config.audio_num_prefix_tokens, config.audio_feature_dim)),
        jnp.ones((1, config.audio_num_prefix_tokens)),
        deterministic=True,
    )
    config.lr = lr
    config.optimizer = "adamw"
    config.weight_decay = 0.0
    optimizer = get_optimizer(config, create_learning_rate_fn(total_steps, warmup_steps, lr), grad_accum_steps=1)
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
    return model, audio_adapter, state


def _prepare_dataset(config, max_examples):
    train_dataset, _ = load_dataset(config)
    if max_examples is not None:
        train_dataset = train_dataset[:max_examples]
    return train_dataset


def make_train_step(model_apply_fn, encoder_apply_fn, audio_adapter_apply_fn, encoder_params, config, sampling_config, args):
    audio_len = config.audio_num_prefix_tokens
    target_len = config.max_length - config.audio_num_prefix_tokens
    eos_weight = config.eos_loss_weight if args.eos_loss_weight is None else args.eos_loss_weight

    def _masked_mean(x, mask, eps=1e-6):
        mask = mask.astype(x.dtype)
        while mask.ndim < x.ndim:
            mask = mask[..., None]
        return (x * mask).sum() / jnp.maximum(mask.sum() * x.shape[-1], eps)

    def _masked_rms(x, mask):
        return jnp.sqrt(_masked_mean(x * x, mask) + 1e-8)

    def _masked_cosine(a, b, mask):
        mask_f = mask.astype(a.dtype)
        while mask_f.ndim < a.ndim:
            mask_f = mask_f[..., None]
        dot = (a * b * mask_f).sum()
        an = jnp.sqrt(((a * a) * mask_f).sum() + 1e-8)
        bn = jnp.sqrt(((b * b) * mask_f).sum() + 1e-8)
        return dot / jnp.maximum(an * bn, 1e-8)

    def _generate(params, z, t_steps, cond_seq, cond_seq_mask):
        z = restore_cond(z, cond_seq, cond_seq_mask)
        x_pred = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)
        t_pairs = jnp.stack([t_steps[:-2], t_steps[1:-1]], axis=1)

        def _step(carry, t_pair):
            z_now, x_prev = carry
            t_now, t_next = t_pair
            z_next, x_next = _ode_step(
                model_apply_fn=model_apply_fn,
                model_params=params,
                z=z_now,
                t=t_now,
                t_next=t_next,
                x_pred_prev=x_prev,
                config=config,
                cfg_scale=args.cfg_scale,
                self_cond_cfg_scale=args.self_cond_cfg_scale,
                cond_seq=cond_seq,
                cond_seq_mask=cond_seq_mask,
            )
            return (z_next, x_next), x_next

        (z, x_pred), xpred_steps = jax.lax.scan(_step, (z, x_pred), t_pairs)
        z, _ = _ode_step(
            model_apply_fn=model_apply_fn,
            model_params=params,
            z=z,
            t=t_steps[-2],
            t_next=t_steps[-1],
            x_pred_prev=x_pred,
            config=config,
            cfg_scale=args.cfg_scale,
            self_cond_cfg_scale=args.self_cond_cfg_scale,
            cond_seq=cond_seq,
            cond_seq_mask=cond_seq_mask,
        )
        return z, xpred_steps

    @jax.jit
    def train_step(state, batch, rng):
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        batch_size = batch["input_ids"].shape[0]
        audio_mask = batch["audio_feature_mask"]
        text_mask = batch["attention_mask"]
        target_attention_mask = text_mask[:, None, :] * text_mask[:, :, None]
        teacher_text = encode_text(
            input_ids=batch["input_ids"],
            attention_mask=target_attention_mask,
            encoder_apply_fn=encoder_apply_fn,
            encoder_params=encoder_params,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
        )
        prefix_targets = jnp.zeros((batch_size, audio_len), dtype=batch["input_ids"].dtype)
        decoder_targets = jnp.concatenate([prefix_targets, batch["input_ids"]], axis=1)
        decoder_loss_mask = jnp.concatenate([jnp.zeros_like(audio_mask), text_mask], axis=1)
        noise_rng, t_rng, dropout_rng = jax.random.split(rng, 3)
        t_steps = get_sampling_steps(
            t_rng,
            n_steps=args.num_sampling_steps,
            time_schedule=sampling_config.time_schedule,
            P_mean=config.denoiser_p_mean,
            P_std=config.denoiser_p_std,
        )

        def loss_fn(trainable):
            audio_prefix = audio_adapter_apply_fn(
                {"params": trainable["audio_adapter"]},
                batch["audio_features"],
                audio_mask,
                deterministic=True,
            )
            cond_seq = jnp.concatenate(
                [audio_prefix, jnp.zeros((batch_size, target_len, teacher_text.shape[-1]), dtype=teacher_text.dtype)],
                axis=1,
            )
            cond_mask_2d = jnp.concatenate([audio_mask, jnp.zeros_like(text_mask)], axis=1)
            noise = jax.random.normal(
                noise_rng,
                (batch_size, config.max_length, teacher_text.shape[-1]),
                dtype=jnp.float32,
            ) * config.denoiser_noise_scale
            sampled, xpred_steps = _generate(trainable["elf"], noise, t_steps, cond_seq, cond_mask_2d)
            sampled_text = sampled[:, audio_len:]
            mse = _masked_mean((sampled_text - teacher_text) ** 2, text_mask)
            cosine = _masked_cosine(sampled_text, teacher_text, text_mask)
            cos_loss = 1.0 - cosine
            sampled_rms = _masked_rms(sampled_text, text_mask)
            teacher_rms = jax.lax.stop_gradient(_masked_rms(teacher_text, text_mask))
            rms_ratio = sampled_rms / jnp.maximum(teacher_rms, 1e-6)
            rms_loss = (jnp.log(jnp.maximum(rms_ratio, 1e-6))) ** 2
            sampled_mean = _masked_mean(sampled_text, text_mask)
            teacher_mean = jax.lax.stop_gradient(_masked_mean(teacher_text, text_mask))
            mean_loss = (sampled_mean - teacher_mean) ** 2
            step_xpred_mse = jnp.zeros((), dtype=sampled.dtype)
            step_xpred_cos_loss = jnp.zeros((), dtype=sampled.dtype)
            step_xpred_rms_loss = jnp.zeros((), dtype=sampled.dtype)
            if (
                args.step_xpred_mse_weight > 0
                or args.step_xpred_cos_weight > 0
                or args.step_xpred_rms_weight > 0
            ):
                xpred_text = xpred_steps[:, :, audio_len:]
                step_mask = jnp.broadcast_to(text_mask[None, ...], xpred_text.shape[:3])
                teacher_steps = jnp.broadcast_to(teacher_text[None, ...], xpred_text.shape)
                step_xpred_mse = _masked_mean((xpred_text - teacher_steps) ** 2, step_mask)
                step_xpred_cosine = _masked_cosine(xpred_text, teacher_steps, step_mask)
                step_xpred_cos_loss = 1.0 - step_xpred_cosine
                step_xpred_rms = _masked_rms(xpred_text, step_mask)
                teacher_steps_rms = jax.lax.stop_gradient(_masked_rms(teacher_steps, step_mask))
                step_xpred_rms_ratio = step_xpred_rms / jnp.maximum(teacher_steps_rms, 1e-6)
                step_xpred_rms_loss = (jnp.log(jnp.maximum(step_xpred_rms_ratio, 1e-6))) ** 2
            decoder_ce = jnp.zeros((), dtype=sampled.dtype)
            artifact_unlikelihood = jnp.zeros((), dtype=sampled.dtype)
            artifact_expected = jnp.zeros((), dtype=sampled.dtype)
            if (
                args.sampled_decoder_ce_weight > 0
                or args.artifact_unlikelihood_weight > 0
                or args.artifact_expected_weight > 0
            ):
                decoder_latent = jax.lax.stop_gradient(sampled) if args.sampled_decoder_stop_gradient else sampled
                decoder_t = jnp.ones((batch_size,), dtype=sampled.dtype)
                decoder_input = (
                    jnp.concatenate([decoder_latent, jnp.zeros_like(decoder_latent)], axis=-1)
                    if config.self_cond_prob > 0 else decoder_latent
                )
                self_cond_cfg_scale = (
                    jnp.full((batch_size,), args.self_cond_cfg_scale, dtype=sampled.dtype)
                    if config.num_self_cond_cfg_tokens > 0 else None
                )
                _, decoder_logits = model_apply_fn(
                    {"params": trainable["elf"]},
                    decoder_input,
                    decoder_t,
                    deterministic=False,
                    rngs={"dropout": dropout_rng},
                    self_cond_cfg_scale=self_cond_cfg_scale,
                    decoder_step_active=jnp.array(True),
                )
                log_probs = jax.nn.log_softmax(decoder_logits.astype(jnp.float32), axis=-1)
                probs_full = jnp.exp(log_probs)
                if args.sampled_decoder_ce_weight > 0:
                    ce = -jnp.take_along_axis(
                        log_probs,
                        decoder_targets[..., None],
                        axis=-1,
                    ).squeeze(-1)
                    ce_weight = jnp.ones_like(ce)
                    if config.eos_token_id is not None and eos_weight != 1.0:
                        ce_weight = jnp.where(
                            decoder_targets == config.eos_token_id,
                            jnp.asarray(eos_weight, dtype=ce.dtype),
                            ce_weight,
                        )
                    weighted_mask = decoder_loss_mask * ce_weight
                    decoder_ce = (ce * weighted_mask).sum() / jnp.maximum(weighted_mask.sum(), 1.0)
                if args.artifact_unlikelihood_weight > 0:
                    pred_ids = jax.lax.stop_gradient(jnp.argmax(decoder_logits, axis=-1))
                    text_pred = pred_ids[:, audio_len:]
                    text_log_probs = log_probs[:, audio_len:]
                    pos_mask = jnp.ones_like(text_pred, dtype=text_log_probs.dtype)
                    # T5-small tokenization in this repo maps common artifacts as:
                    # "and a" -> [11, 3, 9], "ing a" -> [3, 53, 3, 9],
                    # "aa" -> [3, 9, 9], "ss" -> [3, 7, 7].
                    and_id = jnp.asarray(11, dtype=text_pred.dtype)
                    space_id = jnp.asarray(3, dtype=text_pred.dtype)
                    a_id = jnp.asarray(9, dtype=text_pred.dtype)
                    ing_id = jnp.asarray(53, dtype=text_pred.dtype)
                    s_id = jnp.asarray(7, dtype=text_pred.dtype)
                    bad_probs = []
                    bad_masks = []
                    if text_pred.shape[1] >= 3:
                        ctx_and_space = (text_pred[:, :-2] == and_id) & (text_pred[:, 1:-1] == space_id)
                        bad_probs.append(jnp.exp(text_log_probs[:, 2:, 9]))
                        bad_masks.append(ctx_and_space.astype(text_log_probs.dtype) * pos_mask[:, 2:])
                        ctx_space_a = (text_pred[:, :-2] == space_id) & (text_pred[:, 1:-1] == a_id)
                        bad_probs.append(jnp.exp(text_log_probs[:, 2:, 9]))
                        bad_masks.append(ctx_space_a.astype(text_log_probs.dtype) * pos_mask[:, 2:])
                        ctx_space_s = (text_pred[:, :-2] == space_id) & (text_pred[:, 1:-1] == s_id)
                        bad_probs.append(jnp.exp(text_log_probs[:, 2:, 7]))
                        bad_masks.append(ctx_space_s.astype(text_log_probs.dtype) * pos_mask[:, 2:])
                    if text_pred.shape[1] >= 4:
                        ctx_ing_a = (
                            (text_pred[:, :-3] == space_id)
                            & (text_pred[:, 1:-2] == ing_id)
                            & (text_pred[:, 2:-1] == space_id)
                        )
                        bad_probs.append(jnp.exp(text_log_probs[:, 3:, 9]))
                        bad_masks.append(ctx_ing_a.astype(text_log_probs.dtype) * pos_mask[:, 3:])
                    if bad_probs:
                        probs = jnp.concatenate([p.reshape(-1) for p in bad_probs], axis=0)
                        masks = jnp.concatenate([m.reshape(-1) for m in bad_masks], axis=0)
                        unlikelihood = -jnp.log(jnp.maximum(1.0 - probs, args.artifact_unlikelihood_margin))
                        artifact_unlikelihood = (unlikelihood * masks).sum() / jnp.maximum(masks.sum(), 1.0)
                if args.artifact_expected_weight > 0:
                    text_probs = probs_full[:, audio_len:]
                    text_valid = jnp.ones(text_probs.shape[:2], dtype=text_probs.dtype)
                    p_and = text_probs[:, :, 11]
                    p_space = text_probs[:, :, 3]
                    p_a = text_probs[:, :, 9]
                    p_ing = text_probs[:, :, 53]
                    p_s = text_probs[:, :, 7]
                    expected_num = jnp.zeros((), dtype=text_probs.dtype)
                    expected_den = jnp.zeros((), dtype=text_probs.dtype)

                    def _position_weight(term_shape, ngram_len):
                        if args.artifact_expected_suffix_boost <= 0:
                            return jnp.ones(term_shape, dtype=text_probs.dtype)
                        end_pos = jnp.arange(term_shape[1], dtype=jnp.int32) + ngram_len
                        suffix_remaining = text_probs.shape[1] - end_pos
                        suffix_mask = suffix_remaining <= args.artifact_expected_suffix_window
                        return 1.0 + args.artifact_expected_suffix_boost * suffix_mask.astype(text_probs.dtype)

                    def _add_expected(term, mask, ngram_len, weight, num, den):
                        weight = jnp.asarray(weight, dtype=term.dtype)
                        pos_weight = _position_weight(term.shape, ngram_len)
                        weighted_mask = mask * pos_weight
                        term = term * weighted_mask
                        return num + weight * term.sum(), den + weighted_mask.sum()

                    if text_probs.shape[1] >= 3:
                        tri_mask = text_valid[:, :-2] * text_valid[:, 1:-1] * text_valid[:, 2:]
                        expected_num, expected_den = _add_expected(
                            p_and[:, :-2] * p_space[:, 1:-1] * p_a[:, 2:],
                            tri_mask,
                            3,
                            args.artifact_expected_and_a_weight,
                            expected_num,
                            expected_den,
                        )
                        expected_num, expected_den = _add_expected(
                            p_space[:, :-2] * p_a[:, 1:-1] * p_a[:, 2:],
                            tri_mask,
                            3,
                            args.artifact_expected_aa_weight,
                            expected_num,
                            expected_den,
                        )
                        expected_num, expected_den = _add_expected(
                            p_space[:, :-2] * p_s[:, 1:-1] * p_s[:, 2:],
                            tri_mask,
                            3,
                            args.artifact_expected_ss_weight,
                            expected_num,
                            expected_den,
                        )
                    if text_probs.shape[1] >= 4:
                        quad_mask = (
                            text_valid[:, :-3]
                            * text_valid[:, 1:-2]
                            * text_valid[:, 2:-1]
                            * text_valid[:, 3:]
                        )
                        expected_num, expected_den = _add_expected(
                            p_space[:, :-3] * p_ing[:, 1:-2] * p_space[:, 2:-1] * p_a[:, 3:],
                            quad_mask,
                            4,
                            args.artifact_expected_ing_a_weight,
                            expected_num,
                            expected_den,
                        )
                    artifact_expected = expected_num / jnp.maximum(expected_den, 1.0)
            loss = (
                args.mse_weight * mse
                + args.cos_weight * cos_loss
                + args.rms_weight * rms_loss
                + args.mean_weight * mean_loss
                + args.step_xpred_mse_weight * step_xpred_mse
                + args.step_xpred_cos_weight * step_xpred_cos_loss
                + args.step_xpred_rms_weight * step_xpred_rms_loss
                + args.sampled_decoder_ce_weight * decoder_ce
                + args.artifact_unlikelihood_weight * artifact_unlikelihood
                + args.artifact_expected_weight * artifact_expected
            )
            return loss, (
                mse, cos_loss, rms_loss, mean_loss, decoder_ce,
                sampled_rms, teacher_rms, rms_ratio, cosine,
                step_xpred_mse, step_xpred_cos_loss, step_xpred_rms_loss,
                artifact_unlikelihood, artifact_expected,
            )

        trainable = {"elf": state.params, "audio_adapter": state.audio_adapter_params}
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(trainable)
        updates, new_opt_state = state.tx.update(grads["elf"], state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        if args.freeze_audio_adapter:
            new_audio_params = state.audio_adapter_params
            new_audio_opt_state = state.audio_adapter_opt_state
        else:
            audio_updates, new_audio_opt_state = state.tx.update(
                grads["audio_adapter"],
                state.audio_adapter_opt_state,
                state.audio_adapter_params,
            )
            new_audio_params = optax.apply_updates(state.audio_adapter_params, audio_updates)
        new_state = state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            audio_adapter_params=new_audio_params,
            audio_adapter_opt_state=new_audio_opt_state,
            ema_params1=jax.tree_util.tree_map(
                lambda e, p: e * config.ema_decay1 + p * (1 - config.ema_decay1),
                state.ema_params1,
                new_params,
            ),
            ema_audio_adapter_params1=jax.tree_util.tree_map(
                lambda e, p: e * config.ema_decay1 + p * (1 - config.ema_decay1),
                state.ema_audio_adapter_params1,
                new_audio_params,
            ),
        )
        (
            mse, cos_loss, rms_loss, mean_loss, decoder_ce,
            sampled_rms, teacher_rms, rms_ratio, cosine,
            step_xpred_mse, step_xpred_cos_loss, step_xpred_rms_loss,
            artifact_unlikelihood, artifact_expected,
        ) = aux
        return new_state, {
            "loss": loss,
            "mse": mse,
            "cos_loss": cos_loss,
            "rms_loss": rms_loss,
            "mean_loss": mean_loss,
            "decoder_ce": decoder_ce,
            "step_xpred_mse": step_xpred_mse,
            "step_xpred_cos_loss": step_xpred_cos_loss,
            "step_xpred_rms_loss": step_xpred_rms_loss,
            "artifact_unlikelihood": artifact_unlikelihood,
            "artifact_expected": artifact_expected,
            "sampled_rms": sampled_rms,
            "teacher_rms": teacher_rms,
            "rms_ratio": rms_ratio,
            "cosine": cosine,
        }

    return train_step


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    sampling_config = load_sampling_configs(config.sampling_configs_path)[0]
    sampling_config.num_sampling_steps = [args.num_sampling_steps]
    sampling_config.cfgs = [args.cfg_scale]
    sampling_config.self_cond_cfg_scales = [args.self_cond_cfg_scale]
    if args.time_schedule is not None:
        sampling_config.time_schedule = args.time_schedule

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    train_dataset = _prepare_dataset(config, args.max_train_examples)
    total_steps = max((len(train_dataset) // args.batch_size) * args.epochs, 1)

    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    rng = jax.random.PRNGKey(args.seed)
    model, audio_adapter, state = _build_state(
        config, encoder_config, tokenizer, rng, args.lr, total_steps, args.warmup_steps
    )
    state, _ = load_checkpoint(args.checkpoint, state)
    if args.use_ema_init:
        state = state.replace(
            params=state.ema_params1,
            audio_adapter_params=state.ema_audio_adapter_params1,
        )
    state = state.replace(
        opt_state=state.tx.init(state.params),
        audio_adapter_opt_state=state.tx.init(state.audio_adapter_params),
        step=jnp.asarray(0, dtype=jnp.int32),
        epoch=0,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "trajectory_manifold_args.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "train_examples": len(train_dataset), "steps": total_steps}, f, indent=2)
        f.write("\n")

    train_step = make_train_step(
        model.apply,
        encoder_model.apply,
        audio_adapter.apply,
        encoder_params,
        config,
        sampling_config,
        args,
    )
    dataloader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        max_seq_length=config.max_length,
        pad_token_id=pad_token_id,
        distributed=False,
        audio_conditioning=True,
        audio_feature_key=config.audio_feature_key,
        audio_num_prefix_tokens=config.audio_num_prefix_tokens,
    )
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, total=len(train_dataset) // args.batch_size, desc=f"epoch {epoch + 1}/{args.epochs}")
        recent = []
        for batch in pbar:
            batch = {k: v for k, v in batch.items() if isinstance(v, np.ndarray)}
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, step_rng)
            global_step += 1
            host_metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
            recent.append(host_metrics)
            if len(recent) > args.log_freq:
                recent.pop(0)
            if global_step == 1 or global_step % args.log_freq == 0:
                avg = {k: sum(m[k] for m in recent) / len(recent) for k in recent[0]}
                pbar.set_postfix({k: f"{v:.4f}" for k, v in avg.items()})
                print(json.dumps({"step": global_step, **avg}, ensure_ascii=False))
            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save_checkpoint(jax_utils.replicate(state), args.output_dir, global_step)
    save_checkpoint(jax_utils.replicate(state), args.output_dir, global_step)
    print(json.dumps({"output_dir": args.output_dir, "steps": global_step}, indent=2))


if __name__ == "__main__":
    main()
