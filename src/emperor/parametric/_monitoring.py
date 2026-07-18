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
        parameter_clip_range = getattr(mixture, "clip_range", None)
        if parameter_clip_range is None or parameter_clip_range <= 0:
            return values.new_zeros(())
        clip_saturation_mask = values.abs() >= float(parameter_clip_range)
        clip_saturation_fraction = clip_saturation_mask.float().mean()
        return clip_saturation_fraction

    @staticmethod
    def router_entropy(
        probabilities: Tensor,
        top_k: int | None = None,
    ) -> Tensor:
        route_probability_count = 1 if top_k == 1 else probabilities.shape[-1]
        flattened_route_probabilities = probabilities.reshape(
            -1, route_probability_count
        ).float()
        route_choice_dimension = -1
        probability_normalization_epsilon = 1e-12
        route_probability_mass = flattened_route_probabilities.sum(
            dim=route_choice_dimension,
            keepdim=True,
        ).clamp_min(probability_normalization_epsilon)
        normalized_route_probabilities = (
            flattened_route_probabilities / route_probability_mass
        )
        log_safe_route_probabilities = normalized_route_probabilities.clamp_min(
            probability_normalization_epsilon
        )
        per_route_entropy = -(
            log_safe_route_probabilities.log() * normalized_route_probabilities
        ).sum(dim=route_choice_dimension)
        mean_route_entropy = per_route_entropy.mean()
        return mean_route_entropy

    @staticmethod
    def utilization(
        probabilities: Tensor | None,
        indices: Tensor | None,
        num_experts: int,
    ) -> Tensor | None:
        if num_experts <= 0:
            return None
        if indices is not None:
            expert_selection_counts = torch.zeros(num_experts, device=indices.device)
            flattened_expert_indices = indices.detach().long().reshape(-1)
            valid_expert_index_mask = (flattened_expert_indices >= 0) & (
                flattened_expert_indices < num_experts
            )
            if valid_expert_index_mask.any():
                expert_dimension = 0
                expert_selection_counts.scatter_add_(
                    expert_dimension,
                    flattened_expert_indices[valid_expert_index_mask],
                    torch.ones_like(
                        flattened_expert_indices[valid_expert_index_mask],
                        dtype=expert_selection_counts.dtype,
                    ),
                )
            total_expert_selections = expert_selection_counts.sum().clamp_min(1.0)
            expert_utilization = expert_selection_counts / total_expert_selections
            return expert_utilization
        if probabilities is None or probabilities.shape[-1] != num_experts:
            return None
        flattened_expert_probabilities = (
            probabilities.detach().float().reshape(-1, num_experts)
        )
        expert_dimension = -1
        probability_normalization_epsilon = 1e-12
        expert_probability_mass = flattened_expert_probabilities.sum(
            dim=expert_dimension,
            keepdim=True,
        ).clamp_min(probability_normalization_epsilon)
        normalized_expert_probabilities = (
            flattened_expert_probabilities / expert_probability_mass
        )
        routing_input_dimension = 0
        mean_expert_utilization = normalized_expert_probabilities.mean(
            dim=routing_input_dimension
        )
        return mean_expert_utilization


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

        self.__validate_single_monitor(trainer)
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

    @staticmethod
    def __validate_single_monitor(trainer: Trainer) -> None:
        configured_monitor_count = sum(
            isinstance(callback, ParametricLayerMonitorCallback)
            for callback in getattr(trainer, "callbacks", ())
        )
        if configured_monitor_count > 1:
            raise ValueError(
                "Only one ParametricLayerMonitorCallback may be configured per Trainer."
            )

    def __wrap_forward(
        self,
        module_name: str,
        parametric_layer: Module,
        pl_module: LightningModule,
    ) -> None:
        original_forward = parametric_layer.forward

        def monitored_forward(*args: object, **kwargs: object) -> object:
            parametric_layer_id = id(parametric_layer)
            if not self.__should_sample(pl_module):
                self._observations.pop(parametric_layer_id, None)
                return original_forward(*args, **kwargs)
            parametric_observation = _ParametricObservation()
            self._observations[parametric_layer_id] = parametric_observation
            try:
                forward_output = original_forward(*args, **kwargs)
                self.__emit_observation(
                    pl_module,
                    module_name,
                    parametric_layer,
                    parametric_observation,
                )
                return forward_output
            finally:
                self._observations.pop(parametric_layer_id, None)

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
            generated_parameter_output = original_generate_parameters(*args, **kwargs)
            parametric_observation = self._observations.get(id(parametric_layer))
            if parametric_observation is not None:
                self.__record_generated_parameters(
                    parametric_observation,
                    generated_parameter_output,
                )
            return generated_parameter_output

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
        generated_weights, generated_bias, skip_mask, auxiliary_loss = output
        if torch.is_tensor(generated_weights):
            observation.generated_weights = generated_weights.detach()
        if torch.is_tensor(generated_bias):
            observation.generated_bias = generated_bias.detach()
        if torch.is_tensor(skip_mask):
            observation.skip_mask = skip_mask.detach()
        if torch.is_tensor(auxiliary_loss):
            observation.auxiliary_loss = auxiliary_loss.detach()

    def __wrap_affine_callback(self, parametric_layer: Module) -> None:
        affine_callback_method_name = "_compute_affine_transformation_callback"
        original_affine_callback = getattr(
            parametric_layer,
            affine_callback_method_name,
        )

        def monitored_affine_callback(
            *args: object,
            **kwargs: object,
        ) -> object:
            affine_output = original_affine_callback(*args, **kwargs)
            parametric_observation = self._observations.get(id(parametric_layer))
            if parametric_observation is not None:
                affine_input = args[2] if len(args) > 2 else kwargs.get("input")
                if torch.is_tensor(affine_input):
                    parametric_observation.affine_input = affine_input.detach()
                if torch.is_tensor(affine_output):
                    parametric_observation.affine_output = affine_output.detach()
            return affine_output

        self.__install_method_wrapper(
            parametric_layer,
            affine_callback_method_name,
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
            sampling_output = original_sampling_method(*args, **kwargs)
            parametric_observation = self._observations.get(id(parametric_layer))
            if parametric_observation is not None:
                router_sample = self.__parse_router_sample(sampling_output)
                if router_sample is not None:
                    parametric_observation.set_sample(slot, router_sample)
            return sampling_output

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
        detached_probabilities = (
            probabilities.detach() if torch.is_tensor(probabilities) else None
        )
        detached_expert_indices = indices.detach() if torch.is_tensor(indices) else None
        detached_auxiliary_loss = (
            auxiliary_loss.detach() if torch.is_tensor(auxiliary_loss) else None
        )
        return _RouterSample(
            probabilities=detached_probabilities,
            indices=detached_expert_indices,
            auxiliary_loss=detached_auxiliary_loss,
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
        weight_expert_utilization = self.__calculate_utilization(
            parametric_layer,
            observation,
            "weight",
        )
        bias_expert_utilization = self.__calculate_utilization(
            parametric_layer,
            observation,
            "bias",
        )
        logger = getattr(pl_module, "logger", None)
        experiment = getattr(logger, "experiment", None)
        global_step = getattr(pl_module, "global_step", 0)
        tracking_context = _ParametricTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            parametric_layer=parametric_layer,
            observation=observation,
            weight_utilization=weight_expert_utilization,
            bias_utilization=bias_expert_utilization,
            experiment=experiment,
            global_step=global_step,
        )
        self.__track_parametric_diagnostics(tracking_context)

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
        generated_parameter_l2_norm = parameter_values.detach().float().norm()
        context.pl_module.log(
            f"{context.module_name}/parametric/generated_{slot}_norm",
            generated_parameter_l2_norm,
        )

    def __track_clip_saturation_fraction(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        parameter_values = self.__parameter_values(context.observation, slot)
        if parameter_values is None:
            return
        parameter_mixture = self.__parameter_mixture(
            context.parametric_layer,
            slot,
        )
        diagnostic_parameter_values = parameter_values.detach().float()
        clip_saturation_fraction = _ParametricDiagnostics.clip_saturation(
            parameter_mixture,
            diagnostic_parameter_values,
        )
        context.pl_module.log(
            f"{context.module_name}/parametric/{slot}_clip_saturation_fraction",
            clip_saturation_fraction,
        )

    @staticmethod
    def __track_auxiliary_loss(context: _ParametricTrackingContext) -> None:
        auxiliary_loss = context.observation.auxiliary_loss
        if auxiliary_loss is None:
            return
        mean_auxiliary_loss = auxiliary_loss.detach().float().mean()
        context.pl_module.log(
            f"{context.module_name}/parametric/auxiliary_loss",
            mean_auxiliary_loss,
        )

    @staticmethod
    def __track_skip_fraction(context: _ParametricTrackingContext) -> None:
        skip_mask = context.observation.skip_mask
        if skip_mask is None:
            return
        skip_fraction = skip_mask.detach().float().mean()
        context.pl_module.log(
            f"{context.module_name}/parametric/skip_fraction",
            skip_fraction,
        )

    @staticmethod
    def __track_drop_fraction(context: _ParametricTrackingContext) -> None:
        skip_mask = context.observation.skip_mask
        if skip_mask is None:
            return
        skip_fraction = skip_mask.detach().float().mean()
        drop_fraction = 1.0 - skip_fraction
        context.pl_module.log(
            f"{context.module_name}/parametric/drop_fraction",
            drop_fraction,
        )

    @staticmethod
    def __track_affine_output_norm(context: _ParametricTrackingContext) -> None:
        affine_output = context.observation.affine_output
        if affine_output is None:
            return
        affine_output_l2_norm = affine_output.detach().float().norm()
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/output_norm",
            affine_output_l2_norm,
        )

    @staticmethod
    def __track_affine_relative_output_norm(
        context: _ParametricTrackingContext,
    ) -> None:
        affine_input = context.observation.affine_input
        affine_output = context.observation.affine_output
        if affine_input is None or affine_output is None:
            return
        affine_output_l2_norm = affine_output.detach().float().norm()
        stabilized_affine_input_l2_norm = (
            affine_input.detach().float().norm().clamp_min(1e-6)
        )
        relative_affine_output_l2_norm = (
            affine_output_l2_norm / stabilized_affine_input_l2_norm
        )
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/relative_output_norm",
            relative_affine_output_l2_norm,
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
        affine_delta = affine_output.detach().float() - affine_input.detach().float()
        affine_delta_l2_norm = affine_delta.norm()
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/delta_norm",
            affine_delta_l2_norm,
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
        diagnostic_affine_input = affine_input.detach().float()
        affine_delta = affine_output.detach().float() - diagnostic_affine_input
        affine_delta_l2_norm = affine_delta.norm()
        stabilized_affine_input_l2_norm = diagnostic_affine_input.norm().clamp_min(1e-6)
        relative_affine_delta_l2_norm = (
            affine_delta_l2_norm / stabilized_affine_input_l2_norm
        )
        context.pl_module.log(
            f"{context.module_name}/parametric/affine/relative_delta_norm",
            relative_affine_delta_l2_norm,
        )

    @staticmethod
    def __track_router_auxiliary_loss(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        router_sample = context.observation.sample_for(slot)
        if router_sample is None or router_sample.auxiliary_loss is None:
            return
        mean_router_auxiliary_loss = (
            router_sample.auxiliary_loss.detach().float().mean()
        )
        context.pl_module.log(
            f"{context.module_name}/router/{slot}_auxiliary_loss",
            mean_router_auxiliary_loss,
        )

    @staticmethod
    def __track_router_entropy(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        router_sample = context.observation.sample_for(slot)
        if router_sample is None or router_sample.probabilities is None:
            return
        parameter_mixture = ParametricLayerMonitorCallback.__parameter_mixture(
            context.parametric_layer,
            slot,
        )
        diagnostic_route_probabilities = router_sample.probabilities.detach().float()
        router_entropy = _ParametricDiagnostics.router_entropy(
            diagnostic_route_probabilities,
            top_k=getattr(parameter_mixture, "top_k", None),
        )
        context.pl_module.log(
            f"{context.module_name}/router/{slot}_entropy",
            router_entropy,
        )

    @staticmethod
    def __calculate_utilization(
        parametric_layer: Module,
        observation: _ParametricObservation,
        slot: ParametricSlot,
    ) -> Tensor | None:
        router_sample = observation.sample_for(slot)
        if router_sample is None:
            return None
        expert_utilization = _ParametricDiagnostics.utilization(
            router_sample.probabilities,
            router_sample.indices,
            ParametricLayerMonitorCallback.__num_experts(parametric_layer, slot),
        )
        return (
            expert_utilization.detach().float()
            if expert_utilization is not None
            else None
        )

    @staticmethod
    def __num_experts(
        parametric_layer: Module,
        slot: ParametricSlot,
    ) -> int:
        parameter_mixture = ParametricLayerMonitorCallback.__parameter_mixture(
            parametric_layer,
            slot,
        )
        return int(getattr(parameter_mixture, "num_experts", 0) or 0)

    @staticmethod
    def __track_active_slots(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if expert_utilization is None:
            return
        active_slot_count = (expert_utilization > 0).sum().float()
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_active_slots",
            active_slot_count,
        )

    @staticmethod
    def __track_dead_slot_fraction(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if expert_utilization is None:
            return
        dead_slot_mask = expert_utilization <= 0
        dead_slot_fraction = dead_slot_mask.float().mean()
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_dead_slot_fraction",
            dead_slot_fraction,
        )

    @staticmethod
    def __track_maximum_utilization(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if expert_utilization is None:
            return
        maximum_slot_utilization = expert_utilization.max()
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_max_utilization",
            maximum_slot_utilization,
        )

    @staticmethod
    def __track_minimum_utilization(
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if expert_utilization is None:
            return
        minimum_slot_utilization = expert_utilization.min()
        context.pl_module.log(
            f"{context.module_name}/mixture/{slot}_min_utilization",
            minimum_slot_utilization,
        )

    def __track_per_slot_utilization(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if not self.log_per_slot_scalars or expert_utilization is None:
            return
        for slot_index, slot_utilization in enumerate(expert_utilization):
            context.pl_module.log(
                f"{context.module_name}/mixture/{slot}_slot_{slot_index}_utilization",
                slot_utilization,
            )

    def __track_utilization_history(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if expert_utilization is None:
            return
        utilization_history = self._utilization_histories[(context.module_name, slot)]
        utilization_history.append(expert_utilization)

    def __track_utilization_histogram(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if expert_utilization is None or context.experiment is None:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/mixture/histogram/{slot}_utilization",
            expert_utilization,
            context.global_step,
        )

    def __track_utilization_heatmap(
        self,
        context: _ParametricTrackingContext,
        slot: ParametricSlot,
    ) -> None:
        expert_utilization = context.utilization_for(slot)
        if expert_utilization is None or context.experiment is None:
            return
        utilization_history = self._utilization_histories[(context.module_name, slot)]
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.module_name}/mixture/heatmap/{slot}_utilization",
            utilization_history,
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
        for method_replacement in reversed(self._wrapped_methods):
            method_replacement.restore()
        self._wrapped_methods.clear()
        self._observations.clear()
        self._utilization_histories.clear()
        self._emission_policy.clear()
