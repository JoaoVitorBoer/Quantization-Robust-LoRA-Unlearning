import logging
from typing import Any, Iterable

from optimizers.muon import MuonAdamW

logger = logging.getLogger(__name__)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _assert_uniform_group_value(
    groups: Iterable[dict], key: str, expected: Any, config_name: str
):
    normalized_expected = _normalize_value(expected)
    actual_values = {_normalize_value(group[key]) for group in groups}
    if actual_values != {normalized_expected}:
        raise RuntimeError(
            f"Muon validation failed: expected {config_name}={expected}, "
            f"but optimizer groups contained {sorted(actual_values)}"
        )


def _assert_decay_group_value(
    groups: Iterable[dict], key: str, expected: Any, config_name: str
):
    normalized_expected = _normalize_value(expected)
    actual_values = {_normalize_value(group[key]) for group in groups}
    invalid_values = actual_values - {normalized_expected, 0.0}
    if invalid_values or normalized_expected not in actual_values:
        raise RuntimeError(
            f"Muon validation failed: expected {config_name}={expected} on decay groups, "
            f"but optimizer groups contained {sorted(actual_values)}"
        )


def validate_muon_unlearning_run(trainer, mode: str):
    if mode != "unlearn" or getattr(trainer, "optimizer_name", None) != "muon":
        return

    optimizer = trainer.create_optimizer()
    if not isinstance(optimizer, MuonAdamW):
        raise RuntimeError(
            f"Muon validation failed: expected MuonAdamW, got {type(optimizer).__name__}"
        )
    if optimizer.muon_optim is None or not optimizer.muon_params:
        raise RuntimeError(
            "Muon validation failed: no parameters were routed to the Muon optimizer."
        )

    optimizer_kwargs = dict(getattr(trainer, "optimizer_kwargs", {}) or {})

    muon_checks = {
        "muon_lr": "lr",
        "muon_momentum": "momentum",
        "muon_ns_steps": "ns_steps",
        "muon_adjust_lr_fn": "adjust_lr_fn",
    }
    for config_name, group_key in muon_checks.items():
        expected = optimizer_kwargs.get(config_name)
        if expected is not None:
            _assert_uniform_group_value(
                optimizer.muon_optim.param_groups, group_key, expected, config_name
            )

    muon_weight_decay = optimizer_kwargs.get("muon_weight_decay")
    if muon_weight_decay is not None:
        _assert_decay_group_value(
            optimizer.muon_optim.param_groups,
            "weight_decay",
            muon_weight_decay,
            "muon_weight_decay",
        )

    if optimizer.adamw_optim is not None:
        adamw_checks = {
            "adamw_lr": "lr",
            "adamw_betas": "betas",
            "adamw_eps": "eps",
        }
        for config_name, group_key in adamw_checks.items():
            expected = optimizer_kwargs.get(config_name)
            if expected is not None:
                _assert_uniform_group_value(
                    optimizer.adamw_optim.param_groups,
                    group_key,
                    expected,
                    config_name,
                )

        adamw_weight_decay = optimizer_kwargs.get("adamw_weight_decay")
        if adamw_weight_decay is not None:
            _assert_decay_group_value(
                optimizer.adamw_optim.param_groups,
                "weight_decay",
                adamw_weight_decay,
                "adamw_weight_decay",
            )

    logger.info(
        "Muon preflight passed: muon_params=%s adamw_params=%s muon_groups=%s adamw_groups=%s",
        len(optimizer.muon_params),
        len(optimizer.adamw_params),
        len(optimizer.muon_optim.param_groups) if optimizer.muon_optim else 0,
        len(optimizer.adamw_optim.param_groups) if optimizer.adamw_optim else 0,
    )
