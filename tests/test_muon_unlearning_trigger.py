import importlib.machinery
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_deepspeed_stub():
    if "deepspeed" in sys.modules:
        return

    deepspeed = types.ModuleType("deepspeed")
    deepspeed.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)

    class DeepSpeedEngine:
        pass

    deepspeed.DeepSpeedEngine = DeepSpeedEngine
    sys.modules["deepspeed"] = deepspeed


_install_deepspeed_stub()

from optimizers.muon import MuonAdamW
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.muon_validation import validate_muon_unlearning_run


class TinyMuonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


def _named_parameter_group(model: nn.Module) -> list[dict]:
    names, params = zip(*list(model.named_parameters()))
    return [{"params": list(params), "param_names": list(names)}]


def _bare_unlearn_trainer(
    model: nn.Module,
    optimizer_kwargs: dict,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
) -> UnlearnTrainer:
    trainer = UnlearnTrainer.__new__(UnlearnTrainer)
    trainer.model = model
    trainer.model_wrapped = model
    trainer.optimizer = None
    trainer.optimizer_name = "muon"
    trainer.optimizer_kwargs = dict(optimizer_kwargs)
    trainer.args = SimpleNamespace(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return trainer


def _write_executable(path: Path, contents: str):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _read_json_lines(path: Path) -> list[list[str]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_script_with_stubs(tmp_path: Path, script_path: str) -> tuple[list[list[str]], list[list[str]]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    accelerate_log = tmp_path / "accelerate.jsonl"
    python_log = tmp_path / "python.jsonl"

    accelerate_stub = """#!/usr/bin/env bash
set -euo pipefail
"$TEST_REAL_PYTHON" - "$TEST_ACCEL_LOG" "$@" <<'PY'
import json
import sys

path = sys.argv[1]
args = sys.argv[2:]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
PY
"""
    python_stub = """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" || "${1:-}" == "-c" ]]; then
  exec "$TEST_REAL_PYTHON" "$@"
fi
"$TEST_REAL_PYTHON" - "$TEST_PYTHON_LOG" "$@" <<'PY'
import json
import sys

path = sys.argv[1]
args = sys.argv[2:]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
PY
"""

    _write_executable(bin_dir / "accelerate", accelerate_stub)
    _write_executable(bin_dir / "python", python_stub)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["TEST_REAL_PYTHON"] = sys.executable
    env["TEST_ACCEL_LOG"] = str(accelerate_log)
    env["TEST_PYTHON_LOG"] = str(python_log)

    subprocess.run(
        ["bash", str(ROOT / script_path)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    return _read_json_lines(accelerate_log), _read_json_lines(python_log)


def _assert_has_overrides(args: list[str], expected_entries: list[str]):
    for entry in expected_entries:
        assert entry in args, f"Missing CLI override: {entry}\nArgs: {args}"


def test_muonadamw_splits_params_and_steps_both_suboptimizers():
    model = TinyMuonModel()
    optimizer = MuonAdamW(
        _named_parameter_group(model),
        lr=1e-3,
        muon_momentum=0.91,
        muon_ns_steps=7,
        muon_adjust_lr_fn="match_rms_adamw",
        adamw_betas=(0.8, 0.9),
        adamw_eps=1e-6,
    )

    assert optimizer.muon_optim is not None
    assert optimizer.adamw_optim is not None
    assert any(param is model.linear.weight for param in optimizer.muon_params)
    assert all(param is not model.embed.weight for param in optimizer.muon_params)
    assert all(param is not model.lm_head.weight for param in optimizer.muon_params)

    muon_group = optimizer.muon_optim.param_groups[0]
    adamw_group = optimizer.adamw_optim.param_groups[0]

    assert muon_group["lr"] == pytest.approx(1e-3)
    assert muon_group["momentum"] == pytest.approx(0.91)
    assert muon_group["ns_steps"] == 7
    assert muon_group["adjust_lr_fn"] == "match_rms_adamw"
    assert tuple(adamw_group["betas"]) == pytest.approx((0.8, 0.9))
    assert adamw_group["eps"] == pytest.approx(1e-6)

    before_muon = model.linear.weight.detach().clone()
    before_adamw = model.linear.bias.detach().clone()
    for param in model.parameters():
        param.grad = torch.full_like(param, 0.25)

    optimizer.step()

    assert not torch.allclose(model.linear.weight, before_muon)
    assert not torch.allclose(model.linear.bias, before_adamw)


def test_unlearn_trainer_create_optimizer_builds_muon_with_expected_groups():
    model = TinyMuonModel()
    trainer = _bare_unlearn_trainer(
        model=model,
        optimizer_kwargs={
            "muon_momentum": 0.93,
            "muon_ns_steps": 6,
            "muon_adjust_lr_fn": "match_rms_adamw",
            "muon_weight_decay": 0.12,
            "adamw_weight_decay": 0.03,
            "adamw_betas": [0.85, 0.97],
            "adamw_eps": 1e-7,
        },
        learning_rate=2e-4,
        weight_decay=0.2,
    )

    optimizer = trainer.create_optimizer()

    assert isinstance(optimizer, MuonAdamW)
    assert trainer.create_optimizer() is optimizer

    muon_group = optimizer.muon_optim.param_groups[0]
    adamw_weight_decays = sorted(
        group["weight_decay"] for group in optimizer.adamw_optim.param_groups
    )

    assert muon_group["lr"] == pytest.approx(2e-4)
    assert muon_group["momentum"] == pytest.approx(0.93)
    assert muon_group["weight_decay"] == pytest.approx(0.12)
    assert muon_group["ns_steps"] == 6
    assert muon_group["adjust_lr_fn"] == "match_rms_adamw"
    assert adamw_weight_decays == pytest.approx([0.0, 0.03])
    assert any(param is model.linear.weight for param in optimizer.muon_params)
    assert any(param is model.embed.weight for param in optimizer.adamw_params)
    assert any(param is model.linear.bias for param in optimizer.adamw_params)


def test_validate_muon_unlearning_run_checks_real_unlearn_optimizer_wiring():
    model = TinyMuonModel()
    trainer = _bare_unlearn_trainer(
        model=model,
        optimizer_kwargs={
            "muon_momentum": 0.93,
            "muon_ns_steps": 6,
            "muon_adjust_lr_fn": "match_rms_adamw",
            "muon_weight_decay": 0.12,
            "adamw_weight_decay": 0.03,
            "adamw_betas": [0.85, 0.97],
            "adamw_eps": 1e-7,
        },
        learning_rate=2e-4,
        weight_decay=0.2,
    )

    validate_muon_unlearning_run(trainer=trainer, mode="unlearn")

    assert isinstance(trainer.optimizer, MuonAdamW)


@pytest.mark.parametrize(
    ("script_path", "expected_train_calls"),
    [
        ("scripts/muse_unlearn_muon.sh", 2),
        ("scripts/tofu_simnpo_muon.sh", 3),
    ],
)
def test_muon_launch_scripts_pass_expected_optimizer_overrides(
    tmp_path: Path, script_path: str, expected_train_calls: int
):
    accelerate_calls, python_calls = _run_script_with_stubs(tmp_path, script_path)

    assert len(accelerate_calls) == expected_train_calls
    assert len([call for call in python_calls if call and call[0] == "src/eval.py"]) == expected_train_calls

    expected_overrides = [
        "src/train.py",
        "trainer.method_args.optimizer_name=muon",
        "trainer.method_args.optimizer_kwargs.muon_momentum=0.95",
        "trainer.method_args.optimizer_kwargs.muon_ns_steps=5",
        "trainer.method_args.optimizer_kwargs.muon_adjust_lr_fn=match_rms_adamw",
        "trainer.method_args.optimizer_kwargs.adamw_betas=[0.9,0.95]",
        "trainer.method_args.optimizer_kwargs.adamw_eps=1e-8",
    ]

    for call in accelerate_calls:
        _assert_has_overrides(call, expected_overrides)
