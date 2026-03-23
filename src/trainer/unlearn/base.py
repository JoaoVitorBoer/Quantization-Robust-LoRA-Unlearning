from copy import deepcopy
from typing import Any, Optional, Union

import torch
from packaging import version
from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names, nested_detach

from optimizers.muon import MuonAdamW
from trainer.base import FinetuneTrainer


from transformers.utils import (
    is_sagemaker_mp_enabled,
)

from accelerate.utils import (
    is_deepspeed_available,
)

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_only,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_deepspeed_available():
    import deepspeed


class UnlearnTrainer(FinetuneTrainer):
    def __init__(
        self,
        optimizer_name: Optional[str] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        self.optimizer_name = optimizer_name.lower() if optimizer_name else None
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        super().__init__(*args, **kwargs)

    # Adapted from Huggingface DPO Trainer: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _get_decay_parameter_names(self, model):
        if hasattr(self, "get_decay_parameter_names"):
            return set(self.get_decay_parameter_names(model))
        return {
            name
            for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS)
            if "bias" not in name
        }

    def _get_muon_optimizer_grouped_parameters(self, model):
        decay_parameters = self._get_decay_parameter_names(model)
        decay_muon_weight_decay = self.optimizer_kwargs.get(
            "muon_weight_decay", self.args.weight_decay
        )
        decay_adamw_weight_decay = self.optimizer_kwargs.get(
            "adamw_weight_decay", self.args.weight_decay
        )

        decay_group = {
            "params": [],
            "param_names": [],
            "muon_weight_decay": decay_muon_weight_decay,
            "adamw_weight_decay": decay_adamw_weight_decay,
        }
        no_decay_group = {
            "params": [],
            "param_names": [],
            "muon_weight_decay": 0.0,
            "adamw_weight_decay": 0.0,
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            group = decay_group if name in decay_parameters else no_decay_group
            group["params"].append(param)
            group["param_names"].append(name)

        grouped_parameters = [
            group for group in (decay_group, no_decay_group) if group["params"]
        ]
        if not grouped_parameters:
            raise ValueError("Muon optimizer requires at least one trainable parameter.")
        return grouped_parameters

    def create_optimizer(self):
        if self.optimizer_name != "muon":
            return super().create_optimizer()

        if self.optimizer is None:
            opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
            optimizer_kwargs = dict(self.optimizer_kwargs)
            optimizer_kwargs.pop("lr", None)
            optimizer_grouped_parameters = self._get_muon_optimizer_grouped_parameters(
                opt_model
            )
            self.optimizer = MuonAdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                **optimizer_kwargs,
            )

        return self.optimizer

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        The only change to this function is calling the Trainer's compute_loss, as it's often overridden by unlearning methods, and we want to maintain the Trainer's evaluation setup.
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config,
                    "keys_to_ignore_at_inference",
                    ["past_key_values"],
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        ### Call compute_loss of super class since overridden compute_loss is not applicable to eval_dataset.
                        loss, outputs = super().compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
