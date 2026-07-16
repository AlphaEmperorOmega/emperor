from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from lightning.pytorch.callbacks import Callback

from emperor.monitoring import (
    MonitorEmissionPolicy,
    MonitorTensorHistory,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module


ParametricSlot = Literal["weight", "bias"]


@dataclass(frozen=True)
class _MethodReplacement:
    owner: object
    method_name: str
    original_method: Callable[..., object]

    def restore(self) -> None:
        setattr(self.owner, self.method_name, self.original_method)


@dataclass(frozen=True)
class _RouterSample:
    probabilities: Tensor | None
    indices: Tensor | None
    auxiliary_loss: Tensor | None


@dataclass
class _ParametricObservation:
    generated_weights: Tensor | None = None
    generated_bias: Tensor | None = None
    skip_mask: Tensor | None = None
    auxiliary_loss: Tensor | None = None
    affine_input: Tensor | None = None
    affine_output: Tensor | None = None
    weight_sample: _RouterSample | None = None
    bias_sample: _RouterSample | None = None

    def set_sample(self, slot: ParametricSlot, sample: _RouterSample) -> None:
        if slot == "weight":
            self.weight_sample = sample
        else:
            self.bias_sample = sample

    def sample_for(self, slot: ParametricSlot) -> _RouterSample | None:
        return self.weight_sample if slot == "weight" else self.bias_sample


@dataclass(frozen=True)
class _ParametricTrackingContext:
    pl_module: LightningModule
    module_name: str
    parametric_layer: Module
    observation: _ParametricObservation
    weight_utilization: Tensor | None
    bias_utilization: Tensor | None
    experiment: object | None
    global_step: int

    def utilization_for(self, slot: ParametricSlot) -> Tensor | None:
        return self.weight_utilization if slot == "weight" else self.bias_utilization


class _ParametricDiagnostics:
    @staticmethod
    def clip_saturation(mixture: object, values: Tensor) -> Tensor:
        clip_range = getattr(mixture, "clip_range", None)
        if clip_range is None or clip_range <= 0:
            return values.new_zeros(())
        return (values.abs() >= float(clip_range)).float().mean()

    @staticmethod
    def router_entropy(probabilities: Tensor) -> Tensor:
        values = probabilities.reshape(-1, probabilities.shape[-1]).float()
        normalized_values = values / values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        entropy = -(normalized_values.clamp_min(1e-12).log() * normalized_values).sum(
            dim=-1
        )
        return entropy.mean()

    @staticmethod
    def utilization(
        probabilities: Tensor | None,
        indices: Tensor | None,
        num_experts: int,
    ) -> Tensor | None:
        if num_experts <= 0:
            return None
        if indices is not None:
            counts = torch.zeros(num_experts, device=indices.device)
            flat_indices = indices.detach().long().reshape(-1)
            valid_indices = (flat_indices >= 0) & (flat_indices < num_experts)
            if valid_indices.any():
                counts.scatter_add_(
                    0,
                    flat_indices[valid_indices],
                    torch.ones_like(
                        flat_indices[valid_indices],
                        dtype=counts.dtype,
                    ),
                )
            return counts / counts.sum().clamp_min(1.0)
        if probabilities is None or probabilities.shape[-1] != num_experts:
            return None
        values = probabilities.detach().float().reshape(-1, num_experts)
        normalized_mass = values / values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        return normalized_mass.mean(dim=0)


class ParametricLayerMonitorCallback(Callback):
    """Log generated-parameter and mixture-router diagnostics."""

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_slot_scalars: bool = False,
    ) -> None:
        super().__init__()
        self.__validate_positive("log_every_n_steps", log_every_n_steps)
        self.__validate_positive("history_size", history_size)
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_slot_scalars = log_per_slot_scalars
        self._wrapped_methods: list[_MethodReplacement] = []
        self._observations: dict[int, _ParametricObservation] = {}
        self._utilization_histories: dict[
            tuple[str, ParametricSlot],
            MonitorTensorHistory,
        ] = {}
        self._emission_policy = MonitorEmissionPolicy()

    @staticmethod
    def __validate_positive(option_name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{option_name} must be greater than 0.")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.parametric._layer import ParametricLayer

        self.__cleanup()
        for module_name, parametric_layer in pl_module.named_modules():
            if not isinstance(parametric_layer, ParametricLayer):
                continue
            for slot in ("weight", "bias"):
                self._utilization_histories[(module_name, slot)] = MonitorTensorHistory(
                    self.history_size
                )
            self.__wrap_forward(module_name, parametric_layer, pl_module)
            self.__wrap_generate_parameters(parametric_layer)
            self.__wrap_affine_callback(parametric_layer)
            self.__wrap_sampling_methods(parametric_layer)

    def __wrap_forward(
        self,
        module_name: str,
        parametric_layer: Module,
        pl_module: LightningModule,
    ) -> None:
        original_forward = parametric_layer.forward

        def monitored_forward(*args: object, **kwargs: object) -> object:
            layer_id = id(parametric_layer)
            if not self.__should_sample(pl_module):
                self._observations.pop(layer_id, None)
                return original_forward(*args, **kwargs)
            observation = _ParametricObservation()
            self._observations[layer_id] = observation
            output = original_forward(*args, **kwargs)
            self.__emit_observation(
                pl_module,
                module_name,
                parametric_layer,
                observation,
            )
            self._observations.pop(layer_id, None)
            return output

        self.__install_method_wrapper(
            parametric_layer,
            "forward",
            original_forward,
            monitored_forward,
        )

    def __wrap_generate_parameters(self, parametric_layer: Module) -> None:
        original_generate_parameters = parametric_layer._generate_parameters

        def monitored_generate_parameters(
            *args: object,
            **kwargs: object,
        ) -> object:
            output = original_generate_parameters(*args, **kwargs)
            observation = self._observations.get(id(parametric_layer))
            if observation is not None:
                self.__record_generated_parameters(observation, output)
            return output

        self.__install_method_wrapper(
            parametric_layer,
            "_generate_parameters",
            original_generate_parameters,
            monitored_generate_parameters,
        )

    @staticmethod
    def __record_generated_parameters(
        observation: _ParametricObservation,
        output: object,
    ) -> None:
        if not isinstance(output, tuple) or len(output) != 4:
            return
        weights, bias, skip_mask, auxiliary_loss = output
        if torch.is_tensor(weights):
            observation.generated_weights = weights.detach()
        if torch.is_tensor(bias):
            observation.generated_bias = bias.detach()
        if torch.is_tensor(skip_mask):
            observation.skip_mask = skip_mask.detach()
        if torch.is_tensor(auxiliary_loss):
            observation.auxiliary_loss = auxiliary_loss.detach()

    def __wrap_affine_callback(self, parametric_layer: Module) -> None:
        method_name = "_compute_affine_transformation_callback"
        original_affine_callback = getattr(parametric_layer, method_name)

        def monitored_affine_callback(
            *args: object,
            **kwargs: object,
        ) -> object:
            output = original_affine_callback(*args, **kwargs)
            observation = self._observations.get(id(parametric_layer))
            if observation is not None:
                affine_input = args[2] if len(args) > 2 else kwargs.get("input")
                if torch.is_tensor(affine_input):
                    observation.affine_input = affine_input.detach()
                if torch.is_tensor(output):
                    observation.affine_output = output.detach()
            return output

        self.__install_method_wrapper(
            parametric_layer,
            method_name,
            original_affine_callback,
            monitored_affine_callback,
        )

    def __wrap_sampling_methods(self, parametric_layer: Module) -> None:
        sampling_methods: tuple[tuple[str, ParametricSlot], ...] = (
            ("_ParametricLayer__sample_weight_probabilities_and_indices", "weight"),
            ("_ParametricLayer__sample_bias_probabilities_and_indices", "bias"),
        )
        for method_name, slot in sampling_methods:
            if hasattr(parametric_layer, method_name):
                self.__wrap_sampling_method(parametric_layer, method_name, slot)

    def __wrap_sampling_method(
        self,
        parametric_layer: Module,
        method_name: str,
        slot: ParametricSlot,
    ) -> None:
        original_sampling_method = getattr(parametric_layer, method_name)

        def monitored_sampling_method(
            *args: object,
            **kwargs: object,
        ) -> object:
            output = original_sampling_method(*args, **kwargs)
            observation = self._observations.get(id(parametric_layer))
            if observation is not None:
                sample = self.__parse_router_sample(output)
                if sample is not None:
                    observation.set_sample(slot, sample)
            return output

        self.__install_method_wrapper(
            parametric_layer,
            method_name,
            original_sampling_method,
            monitored_sampling_method,
        )

    @staticmethod
    def __parse_router_sample(output: object) -> _RouterSample | None:
        if not isinstance(output, tuple) or len(output) != 4:
            return None
        probabilities, indices, _skip_mask, auxiliary_loss = output
        return _RouterSample(
            probabilities=(
                probabilities.detach() if torch.is_tensor(probabilities) else None
            ),
            indices=indices.detach() if torch.is_tensor(indices) else None,
            auxiliary_loss=(
                auxiliary_loss.detach() if torch.is_tensor(auxiliary_loss) else None
            ),
        )

    def __install_method_wrapper(
        self,
        owner: object,
        method_name: str,
        original_method: Callable[..., object],
        wrapper: Callable[..., object],
    ) -> None:
        setattr(owner, method_name, wrapper)
        self._wrapped_methods.append(
            _MethodReplacement(owner, method_name, original_method)
        )

    def __should_sample(self, pl_module: LightningModule) -> bool:
        global_step = getattr(pl_module, "global_step", 0)
        return global_step % self.log_every_n_steps == 0

    def __emit_observation(
        self,
        pl_module: LightningModule,
        module_name: str,
        parametric_layer: Module,
        observation: _ParametricObservation,
    ) -> None:
        context = _ParametricTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            parametric_layer=parametric_layer,
            observation=observation,
            weight_utilization=self.__calculate_utilization(
                parametric_layer,
                observation,
                "weight",
            ),
            bias_utilization=self.__calculate_utilization(
                parametric_layer,
                observation,
                "bias",
            ),
            experiment=getattr(
                getattr(pl_module, "logger", None),
                "experiment",
                None,
            ),
            global_step=getattr(pl_module, "global_step", 0),
        )
        self.__track_parametric_diagnostics(context)

    def __track_parametric_diagnostics(
        self,
        context: _ParametricTrackingContext,
    ) -> None:
        self.__track_generated_parameter_norm(context, "weight")
        self.__track_generated_parameter_norm(context, "bias")
        self.__track_clip_saturation_fraction(context, "weight")
        self.__track_clip_saturation_fraction(context, "bias")
        self.__track_auxiliary_loss(context)
        self.__track_skip_fraction(context)
        self.__track_drop_fraction(context)
        self.__track_affine_output_norm(context)
        self.__track_affine_relative_output_norm(context)
        self.__track_affine_delta_norm(context)
        self.__track_affine_relative_delta_norm(context)
        self.__track_router_auxiliary_loss(context, "weight")
        self.__track_router_entropy(context, "weight")
        self.__track_active_slots(context, "weight")
        self.__track_dead_slot_fraction(context, "weight")
        self.__track_maximum_utilization(context, "weight")
        self.__track_minimum_utilization(context, "weight")
        self.__track_per_slot_utilization(context, "weight")
        self.__track_utilization_history(context, "weight")
        self.__track_utilization_histogram(context, "weight")
        self.__track_utilization_heatmap(context, "weight")
        self.__track_router_auxiliary_loss(context, "bias")
        self.__track_router_entropy(context, "bias")
        self.__track_active_slots(context, "bias")
        self.__track_dead_slot_fraction(context, "bias")
        self.__track_maximum_utilization(context, "bias")
        self.__track_minimum_utilization(context, "bias")
        self.__track_per_slot_utilization(context, "bias")
        self.__track_utilization_history(context, "bias")
        self.__track_utilization_histogram(context, "bias")
        self.__track_utilization_heatmap(context, "bias")

    @staticmethod
    def __parameter_mixture(
        parametric_layer: Module,
        slot: ParametricSlot,
    ) -> object:
        return (
            parametric_layer.weight_mixture_model
            if slot == "weight"
            else parametric_layer.bias_mixture_model
        )

    @staticmethod
    def __parameter_values(
        observation: _ParametricObservation,
        slot: ParametricSlot,
    ) -> Tensor | None:
        return (
            observation.generated_weights
            if slot == "weight"
            else observation.generated_bias
        )

    def __track_generated_parameter_norm(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        parameter_values = self.__parameter_values(context.observation, slot)
        if parameter_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/generated_{slot}_norm",
            parameter_values.detach().float().norm(),
        )

    def __track_clip_saturation_fraction(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        parameter_values = self.__parameter_values(context.observation, slot)
        if parameter_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/{slot}_clip_saturation_fraction",
            _ParametricDiagnostics.clip_saturation(
                self.__parameter_mixture(context.parametric_layer, slot),
                parameter_values.detach().float(),
            ),
        )

    @staticmethod
    def __track_auxiliary_loss(context: _ParametricTrackingContext) -> None:
        auxiliary_loss = context.observation.auxiliary_loss
        if auxiliary_loss is None:
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/auxiliary_loss",
            auxiliary_loss.detach().float().mean(),
        )

    @staticmethod
    def __track_skip_fraction(context: _ParametricTrackingContext) -> None:
        skip_mask = context.observation.skip_mask
        if skip_mask is None:
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/skip_fraction",
            skip_mask.detach().float().mean(),
        )

    @staticmethod
    def __track_drop_fraction(context: _ParametricTrackingContext) -> None:
        skip_mask = context.observation.skip_mask
        if skip_mask is None:
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/drop_fraction",
            1.0 - skip_mask.detach().float().mean(),
        )

    @staticmethod
    def __track_affine_output_norm(context: _ParametricTrackingContext) -> None:
        affine_output = context.observation.affine_output
        if affine_output is None:
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/output_norm",
            affine_output.detach().float().norm(),
        )

    @staticmethod
    def __track_affine_relative_output_norm(
        context: _ParametricTrackingContext,
    ) -> None:
        affine_input = context.observation.affine_input
        affine_output = context.observation.affine_output
        if affine_input is None or affine_output is None:
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/relative_output_norm",
            affine_output.detach().float().norm()
            / affine_input.detach().float().norm().clamp_min(1e-6),
        )

    @staticmethod
    def __track_affine_delta_norm(context: _ParametricTrackingContext) -> None:
        affine_input = context.observation.affine_input
        affine_output = context.observation.affine_output
        if (
            affine_input is None
            or affine_output is None
            or affine_input.shape != affine_output.shape
        ):
            return
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/delta_norm",
            (affine_output.detach().float() - affine_input.detach().float()).norm(),
        )

    @staticmethod
    def __track_affine_relative_delta_norm(
        context: _ParametricTrackingContext,
    ) -> None:
        affine_input = context.observation.affine_input
        affine_output = context.observation.affine_output
        if (
            affine_input is None
            or affine_output is None
            or affine_input.shape != affine_output.shape
        ):
            return
        input_values = affine_input.detach().float()
        delta_norm = (affine_output.detach().float() - input_values).norm()
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/relative_delta_norm",
            delta_norm / input_values.norm().clamp_min(1e-6),
        )

    @staticmethod
    def __track_router_auxiliary_loss(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        sample = context.observation.sample_for(slot)
        if sample is None or sample.auxiliary_loss is None:
            return
        context.pl_module.log(
            f"{context.module_name}/router/{slot}_auxiliary_loss",
            sample.auxiliary_loss.detach().float().mean(),
        )

    @staticmethod
    def __track_router_entropy(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        sample = context.observation.sample_for(slot)
        if sample is None or sample.probabilities is None:
            return
        context.pl_module.log(
            f"{context.module_name}/router/{slot}_entropy",
            _ParametricDiagnostics.router_entropy(
                sample.probabilities.detach().float()
            ),
        )

    @staticmethod
    def __calculate_utilization(
        parametric_layer: Module,
        observation: _ParametricObservation,
        slot: ParametricSlot,
    ) -> Tensor | None:
        sample = observation.sample_for(slot)
        if sample is None:
            return None
        utilization = _ParametricDiagnostics.utilization(
            sample.probabilities,
            sample.indices,
            ParametricLayerMonitorCallback.__num_experts(parametric_layer, slot),
        )
        return utilization.detach().float() if utilization is not None else None

    @staticmethod
    def __num_experts(
        parametric_layer: Module,
        slot: ParametricSlot,
    ) -> int:
        mixture = ParametricLayerMonitorCallback.__parameter_mixture(
            parametric_layer,
            slot,
        )
        return int(getattr(mixture, "num_experts", 0) or 0)

    @staticmethod
    def __track_active_slots(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if utilization is None:
            return
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_active_slots",
            (utilization > 0).sum().float(),
        )

    @staticmethod
    def __track_dead_slot_fraction(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if utilization is None:
            return
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_dead_slot_fraction",
            (utilization <= 0).float().mean(),
        )

    @staticmethod
    def __track_maximum_utilization(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if utilization is None:
            return
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_max_utilization",
            utilization.max(),
        )

    @staticmethod
    def __track_minimum_utilization(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if utilization is None:
            return
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_min_utilization",
            utilization.min(),
        )

    def __track_per_slot_utilization(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if not self.log_per_slot_scalars or utilization is None:
            return
        for slot_index, slot_utilization in enumerate(utilization):
            context.pl_module.log(
                f"{context.module_name}/mixture/{slot}_slot_{slot_index}_utilization",
                slot_utilization,
            )

    def __track_utilization_history(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if utilization is None:
            return
        self._utilization_histories[(context.module_name, slot)].append(utilization)

    def __track_utilization_histogram(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if utilization is None or context.experiment is None:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/mixture/histogram/{slot}_utilization",
            utilization,
            context.global_step,
        )

    def __track_utilization_heatmap(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        utilization = context.utilization_for(slot)
        if utilization is None or context.experiment is None:
            return
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.module_name}/mixture/heatmap/{slot}_utilization",
            self._utilization_histories[(context.module_name, slot)],
            context.global_step,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self.__cleanup()

    def __cleanup(self) -> None:
        for replacement in reversed(self._wrapped_methods):
            replacement.restore()
        self._wrapped_methods.clear()
        self._observations.clear()
        self._utilization_histories.clear()
        self._emission_policy.clear()
