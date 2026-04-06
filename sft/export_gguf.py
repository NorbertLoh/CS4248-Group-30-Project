import argparse
import os

from unsloth import FastVisionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained Unsloth adapter to GGUF")
    parser.add_argument(
        "--lora-dir",
        default=os.getenv("LORA_DIR", "judge-qwen3-lora"),
        help="Path to the trained adapter directory (default: judge-qwen3-lora)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.getenv("GGUF_OUT_DIR", "judge-qwen3-lora-gguf"),
        help="Output directory for GGUF files (default: judge-qwen3-lora-gguf)",
    )
    parser.add_argument(
        "--quant",
        default=os.getenv("GGUF_QUANT", "q4_k_m"),
        help="GGUF quantization method (default: q4_k_m)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.lora_dir):
        raise FileNotFoundError(f"LoRA directory not found: {args.lora_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading adapter from: {args.lora_dir}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.lora_dir,
        load_in_4bit=True,
    )

    print(f"Exporting GGUF to: {args.out_dir} (quant={args.quant})")
    try:
        model.save_pretrained_gguf(
            args.out_dir,
            tokenizer,
            quantization_method=args.quant,
        )
    except Exception as e:
        raise RuntimeError(
            "GGUF export failed. If this is on remote cluster, rerun this script on a machine with "
            "working Unsloth llama.cpp conversion dependencies/network access."
        ) from e

    print("GGUF export complete.")


if __name__ == "__main__":
    main()