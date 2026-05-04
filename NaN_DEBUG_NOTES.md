# NaN Debug Notes — Additions for `latent_ablation` divergence

These notes describe a small set of additions on top of `1d9c1af gpu + grad accum`
intended to diagnose the NaN you reported around step 20 of
`latent_ablation` (`bs=32, ga=16, pos=neg=32, gen=64`).

**Nothing existing was changed in a behaviour-affecting way.** The new
`debug_finite` path is opt-in (default `False`). The matmul precision change in
`main.py` is the only default-changing edit and is documented inline.

---

## What was added

| File | Change | Default behaviour |
|------|--------|-------------------|
| `main.py` | `os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")` | Default ON (override in shell to disable). |
| `train.py` | `debug_finite` flag through `train_step` and the outer loop, with `_tree_all_finite_jax` / `_assert_host_tree_finite` helpers. | Default OFF. Activates when config has `train.debug_finite: true`. |
| `tools/check_latent_cache.py` | Standalone scanner that walks the latent cache and prints any `.pt` file containing NaN/Inf in `moments` / `moments_flip`. | Manual run. |
| `tools/debug_train_step_synthetic.py` | Toy reproducer that runs `train_step` with synthetic tensors. Confirms the GA path + `debug_finite` plumbing without needing the cache. | Manual run. |

---

## 1. `JAX_DEFAULT_MATMUL_PRECISION=highest` in `main.py`

Verified empirically on this box (4 × A6000):

```
jnp.matmul(2048, 2048) default precision   rel_err vs HIGHEST = 2.93e-04
jnp.matmul(2048, 2048) HIGHEST precision   rel_err vs HIGHEST = 0.00e+00
```

`2.9e-4 ≈ 2^-10` — that is TF32 mantissa precision (≈ bf16). On Ampere/Ada/
Hopper, JAX lowers fp32 matmul to TF32 by default, so the model config's
`attn_fp32: true` does **not** actually buy real fp32 attention on GPU without
this flag. On TPU `attn_fp32: true` is real fp32; this is a GPU-specific gap.

This was **not** sufficient on its own to reproduce the NaN in my testing (see
"What I tried" below), but it is a correctness change worth keeping. Cost is
~1.5–2× slower attention. Override with `JAX_DEFAULT_MATMUL_PRECISION=default`
in the shell if needed for perf comparisons.

## 2. `debug_finite` path in `train.py`

When the YAML has `train.debug_finite: true`, every step:

- runs `_tree_all_finite_jax` on `sg_features`, `gen_samples`, `gen_features`,
  `loss`, `grads`, `state.params`, `new_state.params`
- runs `_assert_host_tree_finite` on the preprocessed image batch and the
  memory-bank `positive_samples` / `negative_samples` before they are fed in
- raises `FloatingPointError` listing the first `finite/*` flag that flips to 0

The `finite/*` flags are also logged into the metrics dict so `metrics.jsonl`
captures the field name when the assertion fires.

This lets us localise the NaN without re-running blind. Expected outcomes:

- `finite/positive_samples` or `finite/negative_samples` → bad `.pt` in cache
- `finite/preprocessed images` (host-side raise) → bad cache file or VAE encode
- `finite/gen_samples` → generator forward overflowed
- `finite/sg_features` / `finite/gen_features` finite=0 with samples finite → MAE forward issue
- `finite/loss` / `finite/grads` with all inputs finite → numerical issue inside `drift_loss`

## 3. `tools/check_latent_cache.py`

```bash
python tools/check_latent_cache.py --root /path/to/latent_cache
```

Walks `train/` and `val/` under the cache root, loads each `.pt`, and asserts
`moments` and `moments_flip` are all finite. Prints up to `--max-report` bad
files, exits non-zero if any are found. Use to rule out cache corruption — this
is currently my top suspect for the step-20 NaN (single-bad-sample pattern).

## 4. `tools/debug_train_step_synthetic.py`

```bash
python tools/debug_train_step_synthetic.py
python tools/debug_train_step_synthetic.py --inject-nan
```

Tiny generator + tiny feature extractor + synthetic `samples`/`negative` tensors
fed into the real `train_step`. The `--inject-nan` flag plants a NaN in
`positive_samples[0,1,2,3,0]`; the host-side guard then raises a
`FloatingPointError` pointing at that index. Useful as a sanity test for the
`debug_finite` plumbing on any GPU box.

---

## What I tried (so we don't repeat work)

Setup: 4 × RTX A6000, conda env with `jax==0.4.37`, this repo at the new commit
above. Scaled reproducer with the **same hyperparameters as your failing run**
(`bs=32, ga=16, micro=2, gen_per_label=64, pos=neg=32, neg_cfg_pw=3.0,
use_bf16=true, attn_fp32=true`, `hf://mae_latent_256`, `hsdp_dim=4` on 4 GPUs)
fed by **256 real ImageNet val latents** I VAE-encoded on the fly.

| step | your trace | my repro (default precision) |
|------|-----------|------------------------------|
| 0 | loss=759, g_norm=1.33e8 | loss=774, g_norm=2.4e7 |
| 10 | loss=768, g_norm=5.17e6 | loss=744, g_norm=6.3e5 |
| 20 | NaN | loss=734, g_norm=8.9e5, finite=OK |
| 29 | NaN | loss=724, g_norm=1.3e6, finite=OK |

g_norm matches in order of magnitude at step 0. Loss decreases monotonically.
**No NaN at 30 steps.** This means I cannot reproduce the divergence at this
scale with fresh VAE-encoded latents, which is consistent with the failure
being either:

1. **Cache contamination** — a few `.pt` files from your offline cache build
   contain non-finite values; once a corrupted sample lands in the memory bank,
   subsequent feature extraction blows up. The step-20 onset (vs step 0)
   matches a delayed sampling probability rather than a numerics problem.
2. **8-GPU FSDP path-specific** — my mesh was `(1, 4)` instead of `(1, 8)`; if
   there is a sharding-related numerical pathology only at 8-way FSDP, I
   wouldn't have hit it.

I'd start with #1 (cheap to verify with `check_latent_cache.py`).

---

## Suggested triage order

1. `python tools/check_latent_cache.py --root <cache>` — 5 min, rules out cache.
2. Set `train.debug_finite: true` in `configs/gen/latent_ablation.yaml`, rerun;
   if it still NaNs, the `FloatingPointError` will name the first non-finite
   tensor. Paste that line back and we'll know exactly where to dig.
3. The matmul precision change is already on by default in `main.py`. If the
   NaN goes away after #1/#2, you can rerun without it
   (`JAX_DEFAULT_MATMUL_PRECISION=default python main.py …`) to see whether the
   precision was load-bearing or just a correctness fix.