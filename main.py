import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--gen", action="store_true", help="Run generator training loop. Default runs MAE training.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    args = parser.parse_args()
    args.output_dir = args.workdir

    # Prefer all visible GPUs on this workstation and avoid slow TPU metadata
    # probes. Users can still override this explicitly in the shell.
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # On Ampere/Ada/Hopper GPUs, JAX's default fp32 matmul lowers to TF32
    # (~bf16 mantissa). The model config sets `attn_fp32: true`, but on GPU
    # that flag alone does not buy real fp32 attention without this. Override
    # in the shell if you want the TF32 speedup back.
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

    # Delay importing train entrypoints until after CLI selection so distributed
    # init only runs for the active path.
    if args.gen:
        from train import main as train_gen_main

        train_gen_main(args)
    else:
        from train_mae import main as train_mae_main

        train_mae_main(args)


if __name__ == "__main__":
    main()
