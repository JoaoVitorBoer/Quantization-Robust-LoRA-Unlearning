#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
            "Compute L2 Frobenius norms of deltaW for MUSE unlearning outputs, per layer.\n"
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
        help="Output CSV path. Defaults to <root-dir>/delta_frobenius_norms_by_layer.csv.",
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
        "--target-layers",
        type=str,
        default=None,
        help=(
            "Optional module-name filter for layer-wise delta calculation. Accepts either a "
            "comma-separated string (e.g. q_proj,v_proj) or a Python/JSON-like list string "
            "(e.g. [\"q_proj\", \"v_proj\"])."
        ),
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


def parse_target_layers(raw_layers: Optional[str]) -> Optional[List[str]]:
    if raw_layers is None:
        return None

    token = raw_layers.strip()
    if not token:
        return None

    if token.startswith("["):
        try:
            parsed = ast.literal_eval(token)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                "Invalid --target-layers list format. Expected something like "
                "['q_proj', 'v_proj']."
            ) from exc
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("Invalid --target-layers value. Expected a list of strings.")
        raw_values = list(parsed)
    else:
        raw_values = token.split(",")

    layers: List[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            raise ValueError("Invalid --target-layers value. Every entry must be a string.")
        layer_name = value.strip()
        if layer_name:
            layers.append(layer_name)

    if not layers:
        raise ValueError("Invalid --target-layers value. No layer names were provided.")

    # Preserve input order while removing duplicates.
    return list(dict.fromkeys(layers))


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
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = False,
):
    return AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=trust_remote_code,
    )


def load_full_ft_model(
    model_dir: Path,
    dtype,
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = False,
):
    return load_hf_model(
        str(model_dir),
        dtype=dtype,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )


def load_merged_lora_model(
    adapter_dir: Path,
    base_model_ref: str,
    dtype,
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = False,
):
    base_model = load_hf_model(
        base_model_ref,
        dtype=dtype,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()
    return merged_model


def frobenius_delta_stats(
    base_model,
    loaded_model,
    target_layers: Optional[List[str]] = None,
) -> Dict[str, float]:
    loaded_params = dict(loaded_model.named_parameters())
    base_keys = []
    loaded_keys = set(loaded_params.keys())

    total_delta_sq = torch.zeros((), dtype=torch.float64)
    total_base_sq = torch.zeros((), dtype=torch.float64)
    matched_param_count = 0
    matched_numel = 0
    target_layer_set = set(target_layers) if target_layers else None

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

            if target_layer_set is not None:
                param_tokens = set(name.split("."))
                if param_tokens.isdisjoint(target_layer_set):
                    continue

            matched_param_count += 1
            base_fp32 = base_param.detach().to(torch.float32)
            loaded_fp32 = loaded_param.detach().to(torch.float32)
            diff = loaded_fp32 - base_fp32
            total_delta_sq += torch.sum(diff * diff, dtype=torch.float64)
            total_base_sq += torch.sum(base_fp32 * base_fp32, dtype=torch.float64)
            matched_numel += base_param.numel()

    extra = loaded_keys - set(base_keys)
    if extra:
        preview = ", ".join(sorted(extra)[:5])
        raise KeyError(
            f"Loaded model has {len(extra)} extra parameters not in base model. "
            f"Examples: {preview}"
        )

    if matched_param_count == 0:
        if target_layers:
            raise ValueError(
                "No parameters matched --target-layers. "
                f"Requested layers: {target_layers}."
            )
        raise ValueError("No parameters available to compute delta norm.")

    if matched_numel <= 0:
        raise ValueError("No tensor elements available to compute delta statistics.")

    delta_l2_frobenius = float(torch.sqrt(total_delta_sq).item())
    base_l2_frobenius = float(torch.sqrt(total_base_sq).item())
    delta_rms = float(torch.sqrt(total_delta_sq / matched_numel).item())
    delta_over_base_l2 = (
        delta_l2_frobenius / base_l2_frobenius if base_l2_frobenius > 0.0 else float("nan")
    )

    return {
        "delta_l2_frobenius": delta_l2_frobenius,
        "base_l2_frobenius": base_l2_frobenius,
        "delta_rms": delta_rms,
        "delta_over_base_l2": delta_over_base_l2,
        "matched_param_count": float(matched_param_count),
        "matched_numel": float(matched_numel),
    }


def infer_layer_names(base_model) -> List[str]:
    layers: List[str] = []
    for name, _ in base_model.named_parameters():
        parts = name.split(".")
        if len(parts) < 2:
            continue
        layer_name = parts[-2]
        if layer_name:
            layers.append(layer_name)

    if not layers:
        raise ValueError("Could not infer layers from model parameter names.")

    # Preserve order while deduplicating.
    return list(dict.fromkeys(layers))


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
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv
        else (root_dir / "delta_frobenius_norms_by_layer.csv")
    )
    output_csv = output_csv.resolve()
    dtype = dtype_from_str(args.dtype)
    target_layers = parse_target_layers(args.target_layers)

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    targets = collect_targets(root_dir)
    if not targets:
        raise RuntimeError(f"No targets found under {root_dir}")
    targets.sort(key=lambda t: (t["split"], t["variant"], t["method"], t["loaded_model"]))

    print(f"Found {len(targets)} targets under {root_dir}")
    if target_layers:
        print(f"Computing per-layer deltas only for selected layers: {target_layers}")
    else:
        print("Computing per-layer deltas for all inferred layers")
    ensure_csv_dir(output_csv)

    rows = []
    current_base_split: Optional[str] = None
    current_base_model_ref: Optional[str] = None
    current_base_model = None
    inferred_layers_by_split: Dict[str, List[str]] = {}

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
                    cache_dir=args.cache_dir,
                    trust_remote_code=args.trust_remote_code,
                )
                current_base_split = split
                current_base_model_ref = base_model_ref

                if target_layers is None:
                    inferred_layers_by_split[split] = infer_layer_names(current_base_model)
                    print(
                        f"  Inferred {len(inferred_layers_by_split[split])} layers for split {split}"
                    )

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
                    cache_dir=args.cache_dir,
                    trust_remote_code=args.trust_remote_code,
                )
            else:
                loaded_model = load_merged_lora_model(
                    Path(loaded_model_ref),
                    base_model_ref=base_model_ref,
                    dtype=dtype,
                    cache_dir=args.cache_dir,
                    trust_remote_code=args.trust_remote_code,
                )

            layers_to_compute = (
                target_layers if target_layers is not None else inferred_layers_by_split[split]
            )

            for layer_name in layers_to_compute:
                print(f"  [debug] Calculating layer: {layer_name}")
                layer_stats = frobenius_delta_stats(
                    base_model,
                    loaded_model,
                    target_layers=[layer_name],
                )
                rows.append(
                    {
                        "variant": variant,
                        "method": method,
                        "split": split,
                        "base_model": base_model_ref,
                        "loaded_model": loaded_model_ref,
                        "layer": layer_name,
                        "delta_l2_frobenius": f"{layer_stats['delta_l2_frobenius']:.10f}",
                        "base_l2_frobenius": f"{layer_stats['base_l2_frobenius']:.10f}",
                        "delta_rms": f"{layer_stats['delta_rms']:.10f}",
                        "delta_over_base_l2": f"{layer_stats['delta_over_base_l2']:.10f}",
                        "matched_param_count": f"{int(layer_stats['matched_param_count'])}",
                        "matched_numel": f"{int(layer_stats['matched_numel'])}",
                    }
                )
                print(
                    "    "
                    f"delta_l2_frobenius: {layer_stats['delta_l2_frobenius']:.10f} | "
                    f"delta_rms: {layer_stats['delta_rms']:.10f} | "
                    f"delta_over_base_l2: {layer_stats['delta_over_base_l2']:.10f}"
                )

            del loaded_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            print(f"  Error: {exc}")
            if args.fail_fast:
                raise

    with output_csv.open("w", newline="") as f:
        fieldnames = [
            "variant",
            "split",
            "method",
            "base_model",
            "loaded_model",
            "layer",
            "delta_l2_frobenius",
            "base_l2_frobenius",
            "delta_rms",
            "delta_over_base_l2",
            "matched_param_count",
            "matched_numel",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {output_csv}")


if __name__ == "__main__":
    main()
