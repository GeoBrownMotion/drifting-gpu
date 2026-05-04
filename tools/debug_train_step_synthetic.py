#!/usr/bin/env python3
"""Smoke-test generator train_step without dataset/checkpoint dependencies."""

import argparse
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import TrainState, _assert_host_tree_finite, train_step
from utils.hsdp_util import set_global_mesh


class TinyGen(nn.Module):
    @nn.compact
    def __call__(self, c, cfg_scale=1.0, train=False):
        del train
        base = self.param("base", nn.initializers.normal(0.02), (4, 4, 1))
        label_w = self.param("label_w", nn.initializers.normal(0.01), (1,))
        cfg_w = self.param("cfg_w", nn.initializers.normal(0.01), (1,))
        c = c.astype(jnp.float32).reshape((-1, 1, 1, 1)) / 1000.0
        cfg_scale = cfg_scale.astype(jnp.float32).reshape((-1, 1, 1, 1))
        samples = jnp.tanh(base[None, ...] + label_w * c + cfg_w * cfg_scale)
        return {"samples": samples}


def feature_apply(params, x, **_kwargs):
    flat = x.reshape((x.shape[0], -1))
    feat = flat @ params["w"] + params["b"]
    return {"toy": feat.reshape((x.shape[0], 2, 3))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pos-per-sample", type=int, default=3)
    parser.add_argument("--neg-per-sample", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--inject-nan", action="store_true")
    args = parser.parse_args()

    set_global_mesh(min(1, jax.local_device_count()))

    batch = args.batch_size
    pos = args.pos_per_sample
    neg = args.neg_per_sample
    height, width, channels = 4, 4, 1
    if batch % args.grad_accum_steps != 0:
        raise ValueError(
            f"batch-size={batch} must be divisible by "
            f"grad-accum-steps={args.grad_accum_steps}"
        )
    labels = jnp.arange(batch, dtype=jnp.int32)
    samples = jnp.linspace(
        -1.0,
        1.0,
        batch * pos * height * width * channels,
        dtype=jnp.float32,
    ).reshape((batch, pos, height, width, channels))
    negative = jnp.linspace(
        0.75,
        -0.75,
        batch * neg * height * width * channels,
        dtype=jnp.float32,
    ).reshape((batch, neg, height, width, channels))
    if args.inject_nan:
        samples = samples.at[0, 1, 2, 3, 0].set(jnp.nan)

    feature_params = {
        "w": jnp.linspace(
            -0.03,
            0.03,
            height * width * channels * 6,
            dtype=jnp.float32,
        ).reshape((height * width * channels, 6)),
        "b": jnp.zeros((6,), dtype=jnp.float32),
    }

    model = TinyGen()
    params = model.init(
        {"params": jax.random.PRNGKey(0)},
        c=jnp.zeros((batch * 2,), dtype=jnp.int32),
        cfg_scale=jnp.ones((batch * 2,), dtype=jnp.float32),
        train=True,
    )["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(1e-3),
        ema_params=params,
        ema_decay=0.9,
    )

    if args.inject_nan:
        try:
            _assert_host_tree_finite("synthetic positive samples", samples, step=7)
        except FloatingPointError as exc:
            print(f"host_guard={exc}")

    step_fn = jax.jit(
        partial(
            train_step,
            rng_init=jax.random.PRNGKey(123),
            learning_rate_fn=lambda step: jnp.asarray(1e-3, dtype=jnp.float32),
            feature_apply=feature_apply,
            activation_kwargs={},
            loss_kwargs={"R_list": (0.02, 0.05)},
            gen_per_label=2,
            cfg_min=1.0,
            cfg_max=1.5,
            neg_cfg_pw=1.0,
            no_cfg_frac=0.0,
            max_grad_norm=2.0,
            grad_accum_steps=args.grad_accum_steps,
            debug_finite=True,
        )
    )
    new_state, metrics = step_fn(state, labels, samples, negative, feature_params)
    jax.block_until_ready(metrics["loss"])

    print(f"devices={jax.devices()}")
    print(f"step={int(state.step)}->{int(new_state.step)}")
    print(f"loss={float(metrics['loss'])}")
    print(f"g_norm={float(metrics['g_norm'])}")
    flags = {k: float(v) for k, v in metrics.items() if k.startswith("finite/")}
    print(f"finite_flags={flags}")


if __name__ == "__main__":
    main()
