#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


MODEL_MARKERS = (
    "config.json",
    "adapter_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute L2 Frobenius norms of deltaW for MUSE unlearning outputs.\n"
            "deltaW is computed against the correct split base model:\n"
            "muse-bench/MUSE-News_target or muse-bench/MUSE-Books_target."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("saves/unlearn/norm_calculation"),
        help="Root directory containing {full_ft,lora}/<method>/<split> folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <root-dir>/delta_frobenius_norms.csv.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache dir.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading models.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error instead of continuing.",
    )
    return parser.parse_args()


def dtype_from_str(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def canonical_split_name(raw_split: str) -> str:
    token = raw_split.strip().lower()
    if token == "news":
        return "News"
    if token == "books":
        return "Books"
    raise ValueError(f"Unknown split folder name: '{raw_split}' (expected NEWS/BOOKS)")


def base_model_for_split(raw_split: str) -> str:
    split = canonical_split_name(raw_split)
    return f"muse-bench/MUSE-{split}_target"


def is_model_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in MODEL_MARKERS)


def latest_checkpoint_dir(path: Path) -> Optional[Path]:
    pattern = re.compile(r"^checkpoint-(\d+)$")
    candidates = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def resolve_model_artifact_dir(path: Path) -> Path:
    if is_model_dir(path):
        return path

    ckpt = latest_checkpoint_dir(path)
    if ckpt is not None and is_model_dir(ckpt):
        return ckpt

    model_like_children = [p for p in path.iterdir() if p.is_dir() and is_model_dir(p)]
    if len(model_like_children) == 1:
        return model_like_children[0]

    raise FileNotFoundError(
        f"Could not find a model artifact directory under '{path}'. "
        "Expected model files in root or inside checkpoint-*."
    )


def load_hf_model(
    model_ref: str,
    dtype,
):
    return AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
    )


def load_full_ft_model(
    model_dir: Path,
    dtype,
):
    return load_hf_model(
        str(model_dir),
        dtype=dtype,
    )


def load_merged_lora_model(
    adapter_dir: Path,
    base_model_ref: str,
    dtype,
):
    base_model = load_hf_model(
        base_model_ref,
        dtype=dtype,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()
    return merged_model


def frobenius_delta_norm(base_model, loaded_model) -> float:
    loaded_params = dict(loaded_model.named_parameters())
    base_keys = []
    loaded_keys = set(loaded_params.keys())

    total_sq = torch.zeros((), dtype=torch.float64)

    with torch.no_grad():
        for name, base_param in base_model.named_parameters():
            base_keys.append(name)
            if name not in loaded_params:
                raise KeyError(f"Parameter '{name}' exists in base model but not loaded model.")
            loaded_param = loaded_params[name]
            if base_param.shape != loaded_param.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"{tuple(base_param.shape)} vs {tuple(loaded_param.shape)}."
                )

            diff = loaded_param.detach().to(torch.float32) - base_param.detach().to(torch.float32)
            total_sq += torch.sum(diff * diff, dtype=torch.float64)

    extra = loaded_keys - set(base_keys)
    if extra:
        preview = ", ".join(sorted(extra)[:5])
        raise KeyError(
            f"Loaded model has {len(extra)} extra parameters not in base model. "
            f"Examples: {preview}"
        )

    return float(torch.sqrt(total_sq).item())


def collect_targets(root_dir: Path) -> List[Dict[str, str]]:
    targets = []
    for variant in ("full_ft", "lora"):
        variant_dir = root_dir / variant
        if not variant_dir.exists():
            continue
        for method_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            for split_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                try:
                    split = canonical_split_name(split_dir.name)
                except ValueError:
                    continue

                base_model = base_model_for_split(split_dir.name)
                artifact_dir = resolve_model_artifact_dir(split_dir)
                targets.append(
                    {
                        "variant": variant,
                        "method": method_dir.name,
                        "split": split,
                        "base_model": base_model,
                        "loaded_model": str(artifact_dir),
                    }
                )
    return targets


def ensure_csv_dir(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    root_dir = args.root_dir.resolve()
    output_csv = args.output_csv.resolve() if args.output_csv else (root_dir / "delta_frobenius_norms.csv")
    output_csv = output_csv.resolve()
    dtype = dtype_from_str(args.dtype)

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    targets = collect_targets(root_dir)
    if not targets:
        raise RuntimeError(f"No targets found under {root_dir}")
    targets.sort(key=lambda t: (t["split"], t["variant"], t["method"], t["loaded_model"]))

    print(f"Found {len(targets)} targets under {root_dir}")
    ensure_csv_dir(output_csv)

    rows = []
    current_base_split: Optional[str] = None
    current_base_model_ref: Optional[str] = None
    current_base_model = None

    for idx, target in enumerate(targets, start=1):
        variant = target["variant"]
        split = target["split"]
        method = target["method"]
        base_model_ref = target["base_model"]
        loaded_model_ref = target["loaded_model"]

        print(f"[{idx}/{len(targets)}] {variant} | {method} | {split}")
        print(f"  base:   {base_model_ref}")
        print(f"  loaded: {loaded_model_ref}")

        try:
            if split != current_base_split:
                if current_base_model is not None:
                    del current_base_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                current_base_model = load_hf_model(
                    base_model_ref,
                    dtype=dtype,
                )
                current_base_split = split
                current_base_model_ref = base_model_ref

            if current_base_model_ref != base_model_ref:
                raise ValueError(
                    f"Inconsistent base model for split {split}: "
                    f"{current_base_model_ref} vs {base_model_ref}"
                )

            base_model = current_base_model

            if variant == "full_ft":
                loaded_model = load_full_ft_model(
                    Path(loaded_model_ref),
                    dtype=dtype,
                )
            else:
                loaded_model = load_merged_lora_model(
                    Path(loaded_model_ref),
                    base_model_ref=base_model_ref,
                    dtype=dtype,
                )

            delta_norm = frobenius_delta_norm(base_model, loaded_model)
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "split": split,
                    "base_model": base_model_ref,
                    "loaded_model": loaded_model_ref,
                    "delta_l2_frobenius": f"{delta_norm:.10f}",
                }
            )
            print(f"  delta_l2_frobenius: {delta_norm:.10f}")

            del loaded_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            raise exc

    with output_csv.open("w", newline="") as f:
        fieldnames = [
            "variant",
            "split",
            "method",
            "base_model",
            "loaded_model",
            "delta_l2_frobenius",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {output_csv}")


if __name__ == "__main__":
    main()
