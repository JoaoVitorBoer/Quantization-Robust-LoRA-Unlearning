import torch
from torch.optim import Optimizer
from typing import Iterable, Dict, Any, Union, List

from .muon import Muon
from .adamw import AdamW


class MuonAdamW(Optimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 0.0003,
        adamw_lr: float = None,
        muon_lr: float = None,
        muon_momentum: float = 0.95,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        muon_weight_decay: float = 0.01,
        adamw_weight_decay: float = 0.01,
        adamw_eps: float = 1e-8,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str = "match_rms_adamw",
    ):
        defaults = dict(
            lr=lr,
            adamw_lr=adamw_lr,
            muon_lr=muon_lr,
            muon_momentum=muon_momentum,
            adamw_betas=adamw_betas,
            muon_weight_decay=muon_weight_decay,
            adamw_weight_decay=adamw_weight_decay,
            adamw_eps=adamw_eps,
            muon_ns_steps=muon_ns_steps,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
        )
        
        super().__init__(params, defaults)

        self.muon_params = []
        self.adamw_params = []

        # Partition parameters into Muon (2D) and AdamW (others)
        # We need to respect parameter groups if provided

        self.muon_groups = []
        self.adamw_groups = []
        self._group_mappings = []
        muon_group_index = 0
        adamw_group_index = 0

        for group in self.param_groups:
            muon_group_params = []
            adamw_group_params = []

            if "param_names" not in group:
                raise ValueError(
                    "MuonAdamW requires named parameters. "
                    "Initialize with model.named_parameters()."
                )
            
            for p_name, p in zip(group["param_names"], group["params"]):
                if p.ndim == 2 and not any(layer in p_name for layer in ["embed", "lm_head", "head"]):
                    muon_group_params.append(p)
                    self.muon_params.append(p)
                else:
                    adamw_group_params.append(p)
                    self.adamw_params.append(p)

            group_mapping = {"outer": group}

            # Muon group config
            # Default lr is global lr if muon_lr is not set
            muon_lr_val = self._get_group_value(group, "muon_lr", "lr")

            muon_group = {
                "params": muon_group_params,
                "lr": muon_lr_val,
                "momentum": group.get("muon_momentum"),
                "weight_decay": group.get("muon_weight_decay"),
                "ns_steps": group.get("muon_ns_steps"),
                "adjust_lr_fn": group.get("muon_adjust_lr_fn"),
            }
            if muon_group_params:
                self.muon_groups.append(muon_group)
                group_mapping["muon_index"] = muon_group_index
                muon_group_index += 1

            # AdamW group config
            # Default lr is global lr if adamw_lr is not set
            adamw_lr_val = self._get_group_value(group, "adamw_lr", "lr")

            adamw_group = {
                "params": adamw_group_params,
                "lr": adamw_lr_val,
                "betas": group.get("adamw_betas"),
                "weight_decay": group.get("adamw_weight_decay"),
                "eps": group.get("adamw_eps"),
            }
            if adamw_group_params:
                self.adamw_groups.append(adamw_group)
                group_mapping["adamw_index"] = adamw_group_index
                adamw_group_index += 1

            self._group_mappings.append(group_mapping)

        self.muon_optim = Muon(self.muon_groups) if self.muon_groups else None
        self.adamw_optim = AdamW(self.adamw_groups) if self.adamw_groups else None
        self._sync_inner_optimizers()

    @staticmethod
    def _get_group_value(group: Dict[str, Any], key: str, fallback_key: str = None):
        value = group.get(key)
        if value is not None:
            return value
        if fallback_key is not None:
            return group.get(fallback_key)
        return None

    def _sync_inner_optimizers(self):
        for group_mapping in self._group_mappings:
            outer_group = group_mapping["outer"]

            if self.muon_optim is not None and "muon_index" in group_mapping:
                muon_group = self.muon_optim.param_groups[group_mapping["muon_index"]]
                muon_group["lr"] = self._get_group_value(outer_group, "muon_lr", "lr")

                for source_key, target_key in (
                    ("muon_momentum", "momentum"),
                    ("muon_weight_decay", "weight_decay"),
                    ("muon_ns_steps", "ns_steps"),
                    ("muon_adjust_lr_fn", "adjust_lr_fn"),
                ):
                    value = outer_group.get(source_key)
                    if value is not None:
                        muon_group[target_key] = value

            if self.adamw_optim is not None and "adamw_index" in group_mapping:
                adamw_group = self.adamw_optim.param_groups[group_mapping["adamw_index"]]
                adamw_group["lr"] = self._get_group_value(outer_group, "adamw_lr", "lr")

                for source_key, target_key in (
                    ("adamw_betas", "betas"),
                    ("adamw_weight_decay", "weight_decay"),
                    ("adamw_eps", "eps"),
                ):
                    value = outer_group.get(source_key)
                    if value is not None:
                        adamw_group[target_key] = value

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._sync_inner_optimizers()

        if self.muon_optim:
            self.muon_optim.step()
        if self.adamw_optim:
            self.adamw_optim.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        if self.muon_optim:
            self.muon_optim.zero_grad(set_to_none=set_to_none)
        if self.adamw_optim:
            self.adamw_optim.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon_optim.state_dict() if self.muon_optim else {},
            "adamw": self.adamw_optim.state_dict() if self.adamw_optim else {},
            # "param_groups": self.param_groups # Helper to inspect groups
        }

    def load_state_dict(self, state_dict):
        if self.muon_optim and "muon" in state_dict:
            self.muon_optim.load_state_dict(state_dict["muon"])
        if self.adamw_optim and "adamw" in state_dict:
            self.adamw_optim.load_state_dict(state_dict["adamw"])
        # super().load_state_dict(state_dict) # This might be tricky with composite state
