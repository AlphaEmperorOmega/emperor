from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.parametric import (
    AdaptiveRouterOptions,
    ClipParameterOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    ParametricLayerConfig,
    ParametricLayerMonitorCallback,
    VectorWeightsMixtureConfig,
)
from emperor.sampler import RouterConfig, SamplerConfig


def _same_bound_method(left: object, right: object) -> bool:
    return getattr(left, "__self__", None) is getattr(
        right,
        "__self__",
        None,
    ) and getattr(left, "__func__", left) is getattr(right, "__func__", right)


def _linear_stack(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=False,
            ),
        ),
    )


def _parametric_layer(*, top_k: int = 2) -> torch.nn.Module:
    input_dim = 2
    output_dim = 2
    num_experts = 2
    mixture_options = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "top_k": top_k,
        "num_experts": num_experts,
        "weighted_parameters_flag": True,
        "clip_parameter_option": ClipParameterOptions.DISABLED,
        "clip_range": 1.0,
    }
    router_config = RouterConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        noisy_topk_flag=False,
        model_config=_linear_stack(input_dim, num_experts),
    )
    layer = ParametricLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        weight_mixture_config=MatrixWeightsMixtureConfig(**mixture_options),
        bias_mixture_config=MatrixBiasMixtureConfig(**mixture_options),
        routing_initialization_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        router_config=router_config,
        sampler_config=SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=top_k == num_experts,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=None,
        ),
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_config=None,
            diagonal_config=None,
            bias_config=None,
            mask_config=None,
            model_config=None,
        ),
    ).build()
    with torch.no_grad():
        layer.weights_router.model[0].model.weight_params.zero_()
        layer.bias_router.model[0].model.weight_params.zero_()
        layer.weight_mixture_model.parameter_bank.copy_(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0, 0.0], [0.0, 2.0]],
                ]
            )
        )
        layer.bias_mixture_model.parameter_bank.copy_(
            torch.tensor(
                [
                    [1.0, -1.0],
                    [3.0, 1.0],
                ]
            )
        )
    return layer


def _weight_only_parametric_layer(
    *,
    input_dim: int,
    output_dim: int,
    top_k: int,
    num_experts: int,
    router_weights: Tensor,
    weight_bank: Tensor,
) -> torch.nn.Module:
    mixture_options = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "top_k": top_k,
        "num_experts": num_experts,
        "weighted_parameters_flag": True,
        "clip_parameter_option": ClipParameterOptions.DISABLED,
        "clip_range": 1.0,
    }
    router_config = RouterConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        noisy_topk_flag=False,
        model_config=_linear_stack(input_dim, num_experts),
    )
    layer = ParametricLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        weight_mixture_config=MatrixWeightsMixtureConfig(**mixture_options),
        bias_mixture_config=None,
        routing_initialization_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        router_config=router_config,
        sampler_config=SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=top_k != 1,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=None,
        ),
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_config=None,
            diagonal_config=None,
            bias_config=None,
            mask_config=None,
            model_config=None,
        ),
    ).build()
    with torch.no_grad():
        layer.weights_router.model[0].model.weight_params.copy_(router_weights)
        layer.weight_mixture_model.parameter_bank.copy_(weight_bank)
    return layer


def _dense_vector_parametric_layer() -> torch.nn.Module:
    input_dim = 2
    output_dim = 2
    num_experts = 4
    layer = ParametricLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        weight_mixture_config=VectorWeightsMixtureConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            top_k=num_experts,
            num_experts=num_experts,
            weighted_parameters_flag=True,
            clip_parameter_option=ClipParameterOptions.DISABLED,
            clip_range=1.0,
        ),
        bias_mixture_config=None,
        routing_initialization_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        router_config=RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=_linear_stack(input_dim, num_experts),
        ),
        sampler_config=SamplerConfig(
            top_k=num_experts,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=True,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=None,
        ),
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_config=None,
            diagonal_config=None,
            bias_config=None,
            mask_config=None,
            model_config=None,
        ),
    ).build()
    identity_rows = torch.tensor(
        [
            [[1.0, 0.0]] * num_experts,
            [[0.0, 1.0]] * num_experts,
        ]
    )
    with torch.no_grad():
        layer.weights_router.parameter_bank.zero_()
        layer.weight_mixture_model.parameter_bank.copy_(identity_rows)
    return layer


class _ParametricTrainingModule(LightningModule):
    def __init__(
        self,
        *,
        fail_after_forward: bool = False,
        learning_rate: float = 0.05,
        parametric_layer: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.parametric = (
            _parametric_layer() if parametric_layer is None else parametric_layer
        )
        self.fail_after_forward = fail_after_forward
        self.learning_rate = learning_rate
        self.logged_calls: list[tuple[int, str, Tensor]] = []
        self.last_input: Tensor | None = None
        self.last_output: Tensor | None = None
        self.last_skip_mask: Tensor | None = None
        self.last_auxiliary_loss: Tensor | None = None

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        (inputs,) = batch
        output, skip_mask, auxiliary_loss = self.parametric(
            input=inputs,
            skip_mask=inputs.new_ones((inputs.shape[0], 1)),
        )
        self.last_input = inputs.detach().clone()
        self.last_output = output.detach().clone()
        self.last_skip_mask = (
            skip_mask.detach().clone() if skip_mask is not None else None
        )
        self.last_auxiliary_loss = auxiliary_loss.detach().clone()
        if self.fail_after_forward:
            raise RuntimeError("deliberate parametric lifecycle failure")
        return output.square().mean() + auxiliary_loss

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if torch.is_tensor(value):
            self.logged_calls.append(
                (int(self.global_step), name, value.detach().clone())
            )
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


class _HistoryObserver(Callback):
    def __init__(self, monitor: ParametricLayerMonitorCallback) -> None:
        super().__init__()
        self.monitor = monitor
        self.lengths: list[dict[tuple[str, str], int]] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        self.lengths.append(
            {
                key: len(history)
                for key, history in self.monitor._utilization_histories.items()
            }
        )


class _CaughtLayerFailureTrainingModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.parametric = _parametric_layer()
        self.caught_errors: list[str] = []

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        (inputs,) = batch
        try:
            self.parametric(inputs.unsqueeze(0))
        except ValueError as error:
            self.caught_errors.append(str(error))
        return self.parametric.weight_mixture_model.parameter_bank.sum() * 0.0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _ObservationObserver(Callback):
    def __init__(self, monitor: ParametricLayerMonitorCallback) -> None:
        super().__init__()
        self.monitor = monitor
        self.counts: list[int] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        self.counts.append(len(self.monitor._observations))


class _UtilizationObserver(Callback):
    def __init__(self, monitor: ParametricLayerMonitorCallback) -> None:
        super().__init__()
        self.monitor = monitor
        self.weight_utilizations: list[Tensor] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        history = self.monitor._utilization_histories[("parametric", "weight")]
        self.weight_utilizations.append(history.tensors[-1].clone())


def _loader(num_batches: int = 1) -> DataLoader:
    values = torch.tensor(
        [
            [1.0, 2.0],
            [-1.0, 3.0],
            [0.5, -2.0],
            [2.0, 1.0],
            [-3.0, 0.25],
            [1.5, -0.5],
        ]
    )
    return DataLoader(
        TensorDataset(values[: num_batches * 2]),
        batch_size=2,
        shuffle=False,
    )


def _trainer(
    root: Path,
    callbacks: list[Callback],
    *,
    logger: bool | TensorBoardLogger = False,
) -> Trainer:
    return Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=root,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


class ParametricMonitorLifecycleTests(unittest.TestCase):
    def assert_logged_close(
        self,
        logged: dict[str, Tensor],
        name: str,
        expected: Tensor | float,
    ) -> None:
        self.assertIn(name, logged)
        torch.testing.assert_close(
            logged[name],
            torch.as_tensor(expected, dtype=logged[name].dtype),
        )
        self.assertTrue(torch.isfinite(logged[name]), name)

    def test_real_trainer_logs_exact_metrics_visuals_and_updates_parameters(
        self,
    ) -> None:
        model = _ParametricTrainingModule()
        monitor = ParametricLayerMonitorCallback(
            log_every_n_steps=1,
            history_size=2,
            log_per_slot_scalars=True,
        )
        layer = model.parametric
        original_forward = layer.forward
        original_generate = layer._generate_parameters
        original_affine = layer._compute_affine_transformation_callback
        initial_weight_bank = layer.weight_mixture_model.parameter_bank.detach().clone()
        initial_bias_bank = layer.bias_mixture_model.parameter_bank.detach().clone()
        initial_weight_router = (
            layer.weights_router.model[0].model.weight_params.detach().clone()
        )
        initial_bias_router = (
            layer.bias_router.model[0].model.weight_params.detach().clone()
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="parametric",
            )
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=_loader())
            tensorboard_logger.experiment.flush()
            events = EventAccumulator(
                tensorboard_logger.log_dir,
                size_guidance={"histograms": 0, "images": 0},
            )
            events.Reload()
            tags = events.Tags()

        expected_input = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        expected_output = torch.tensor([[3.5, 3.0], [0.5, 4.5]])
        torch.testing.assert_close(model.last_input, expected_input)
        torch.testing.assert_close(model.last_output, expected_output)
        torch.testing.assert_close(model.last_skip_mask, torch.ones(2, 1))
        torch.testing.assert_close(model.last_auxiliary_loss, torch.tensor(0.0))

        logged = dict(fit_trainer.logged_metrics)
        input_norm = expected_input.norm()
        output_norm = expected_output.norm()
        delta_norm = (expected_output - expected_input).norm()
        expected_scalars = {
            "parametric/parametric/generated_weight_norm": 3.0,
            "parametric/parametric/generated_bias_norm": math.sqrt(8.0),
            "parametric/parametric/weight_clip_saturation_fraction": 0.5,
            "parametric/parametric/bias_clip_saturation_fraction": 0.5,
            "parametric/parametric/auxiliary_loss": 0.0,
            "parametric/parametric/skip_fraction": 1.0,
            "parametric/parametric/drop_fraction": 0.0,
            "parametric/parametric/affine/output_norm": output_norm,
            "parametric/parametric/affine/relative_output_norm": (
                output_norm / input_norm
            ),
            "parametric/parametric/affine/delta_norm": delta_norm,
            "parametric/parametric/affine/relative_delta_norm": (
                delta_norm / input_norm
            ),
            "parametric/router/weight_auxiliary_loss": 0.0,
            "parametric/router/bias_auxiliary_loss": 0.0,
            "parametric/router/weight_entropy": math.log(2.0),
            "parametric/router/bias_entropy": math.log(2.0),
            "parametric/mixture/weight_active_slots": 2.0,
            "parametric/mixture/bias_active_slots": 2.0,
            "parametric/mixture/weight_dead_slot_fraction": 0.0,
            "parametric/mixture/bias_dead_slot_fraction": 0.0,
            "parametric/mixture/weight_max_utilization": 0.5,
            "parametric/mixture/bias_max_utilization": 0.5,
            "parametric/mixture/weight_min_utilization": 0.5,
            "parametric/mixture/bias_min_utilization": 0.5,
            "parametric/mixture/weight_slot_0_utilization": 0.5,
            "parametric/mixture/weight_slot_1_utilization": 0.5,
            "parametric/mixture/bias_slot_0_utilization": 0.5,
            "parametric/mixture/bias_slot_1_utilization": 0.5,
        }
        for metric_name, expected in expected_scalars.items():
            self.assert_logged_close(logged, metric_name, expected)

        for slot in ("weight", "bias"):
            self.assertIn(
                f"parametric/mixture/histogram/{slot}_utilization",
                tags["histograms"],
            )
            self.assertIn(
                f"parametric/mixture/heatmap/{slot}_utilization",
                tags["images"],
            )

        self.assertFalse(
            torch.equal(
                layer.weight_mixture_model.parameter_bank.detach(),
                initial_weight_bank,
            )
        )
        self.assertFalse(
            torch.equal(
                layer.bias_mixture_model.parameter_bank.detach(),
                initial_bias_bank,
            )
        )
        self.assertFalse(
            torch.equal(
                layer.weights_router.model[0].model.weight_params.detach(),
                initial_weight_router,
            )
        )
        self.assertFalse(
            torch.equal(
                layer.bias_router.model[0].model.weight_params.detach(),
                initial_bias_router,
            )
        )
        self.assertTrue(_same_bound_method(layer.forward, original_forward))
        self.assertTrue(
            _same_bound_method(layer._generate_parameters, original_generate)
        )
        self.assertTrue(
            _same_bound_method(
                layer._compute_affine_transformation_callback,
                original_affine,
            )
        )
        self.assertEqual(monitor._wrapped_methods, [])
        self.assertEqual(monitor._observations, {})
        self.assertEqual(monitor._utilization_histories, {})

    def test_real_trainer_applies_cadence_and_bounded_history(self) -> None:
        model = _ParametricTrainingModule(learning_rate=0.0)
        monitor = ParametricLayerMonitorCallback(
            log_every_n_steps=2,
            history_size=1,
        )
        observer = _HistoryObserver(monitor)
        observation_observer = _ObservationObserver(monitor)

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="parametric-cadence",
            )
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor, observer, observation_observer],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=_loader(num_batches=3))
            tensorboard_logger.experiment.flush()
            events = EventAccumulator(
                tensorboard_logger.log_dir,
                size_guidance={"histograms": 0, "images": 0},
            )
            events.Reload()

        metric_name = "parametric/parametric/generated_weight_norm"
        metric_calls = [
            (step, name) for step, name, _ in model.logged_calls if name == metric_name
        ]
        self.assertEqual(metric_calls, [(0, metric_name), (2, metric_name)])
        self.assertEqual(len(observer.lengths), 3)
        for lengths in observer.lengths:
            self.assertEqual(
                lengths,
                {
                    ("parametric", "weight"): 1,
                    ("parametric", "bias"): 1,
                },
            )
        self.assertEqual(observation_observer.counts, [0, 0, 0])
        self.assertEqual(
            [
                event.step
                for event in events.Histograms(
                    "parametric/mixture/histogram/weight_utilization"
                )
            ],
            [0, 2],
        )
        self.assertEqual(
            [
                event.step
                for event in events.Images(
                    "parametric/mixture/heatmap/weight_utilization"
                )
            ],
            [0, 2],
        )
        self.assertEqual(monitor._wrapped_methods, [])
        self.assertEqual(monitor._observations, {})
        self.assertEqual(monitor._utilization_histories, {})

    def test_real_sparse_weight_only_lifecycle_is_exact_and_asymmetric(
        self,
    ) -> None:
        router_weights = torch.tensor(
            [
                [4.0, 3.0, 0.0, -1.0],
                [0.0, 4.0, 3.0, -1.0],
                [0.0, 3.0, 4.0, -1.0],
            ]
        )
        weight_bank = torch.stack(tuple(torch.eye(3) for _ in range(4)))
        layer = _weight_only_parametric_layer(
            input_dim=3,
            output_dim=3,
            top_k=2,
            num_experts=4,
            router_weights=router_weights,
            weight_bank=weight_bank,
        )
        model = _ParametricTrainingModule(
            learning_rate=0.0,
            parametric_layer=layer,
        )
        monitor = ParametricLayerMonitorCallback(
            log_every_n_steps=1,
            history_size=2,
        )
        observer = _UtilizationObserver(monitor)
        inputs = torch.eye(3)
        loader = DataLoader(TensorDataset(inputs), batch_size=3, shuffle=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="parametric-sparse",
            )
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor, observer],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=loader)
            tensorboard_logger.experiment.flush()
            events = EventAccumulator(tensorboard_logger.log_dir)
            events.Reload()
            tags = events.Tags()

        full_denominator = (
            math.exp(4.0) + math.exp(3.0) + math.exp(0.0) + math.exp(-1.0)
        )
        selected_mass = (math.exp(4.0) + math.exp(3.0)) / full_denominator
        mixture_scale = selected_mass / (selected_mass + 1e-6)
        high_probability = math.exp(4.0) / (math.exp(4.0) + math.exp(3.0))
        low_probability = 1.0 - high_probability
        expected_entropy = -(
            high_probability * math.log(high_probability)
            + low_probability * math.log(low_probability)
        )
        expected_utilization = torch.tensor([1.0 / 6.0, 3.0 / 6.0, 2.0 / 6.0, 0.0])

        torch.testing.assert_close(model.last_output, mixture_scale * inputs)
        self.assertEqual(len(observer.weight_utilizations), 1)
        torch.testing.assert_close(
            observer.weight_utilizations[0],
            expected_utilization,
        )
        logged = dict(fit_trainer.logged_metrics)
        expected_scalars = {
            "parametric/parametric/generated_weight_norm": 3.0 * mixture_scale,
            "parametric/parametric/weight_clip_saturation_fraction": 0.0,
            "parametric/router/weight_auxiliary_loss": 0.0,
            "parametric/router/weight_entropy": expected_entropy,
            "parametric/mixture/weight_active_slots": 3.0,
            "parametric/mixture/weight_dead_slot_fraction": 0.25,
            "parametric/mixture/weight_max_utilization": 0.5,
            "parametric/mixture/weight_min_utilization": 0.0,
        }
        for metric_name, expected in expected_scalars.items():
            self.assert_logged_close(logged, metric_name, expected)

        absent_metrics = {
            "parametric/parametric/generated_bias_norm",
            "parametric/parametric/bias_clip_saturation_fraction",
            "parametric/router/bias_auxiliary_loss",
            "parametric/router/bias_entropy",
            "parametric/mixture/bias_active_slots",
            "parametric/mixture/bias_dead_slot_fraction",
            "parametric/mixture/weight_slot_0_utilization",
        }
        self.assertTrue(absent_metrics.isdisjoint(logged))
        self.assertIn(
            "parametric/mixture/histogram/weight_utilization",
            tags["histograms"],
        )
        self.assertIn(
            "parametric/mixture/heatmap/weight_utilization",
            tags["images"],
        )
        self.assertNotIn(
            "parametric/mixture/histogram/bias_utilization",
            tags["histograms"],
        )
        self.assertNotIn(
            "parametric/mixture/heatmap/bias_utilization",
            tags["images"],
        )

    def test_real_top_one_weight_only_entropy_uses_the_weight_slot(self) -> None:
        router_weights = torch.tensor(
            [
                [2.0, -2.0],
                [-1.0, 1.0],
            ]
        )
        weight_bank = torch.stack((torch.eye(2), 2.0 * torch.eye(2)))
        layer = _weight_only_parametric_layer(
            input_dim=2,
            output_dim=2,
            top_k=1,
            num_experts=2,
            router_weights=router_weights,
            weight_bank=weight_bank,
        )
        model = _ParametricTrainingModule(
            learning_rate=0.0,
            parametric_layer=layer,
        )
        monitor = ParametricLayerMonitorCallback(log_every_n_steps=1)
        inputs = torch.eye(2)
        loader = DataLoader(TensorDataset(inputs), batch_size=2, shuffle=False)
        full_probabilities = torch.softmax(inputs @ router_weights, dim=-1)
        selected_probabilities, selected_indices = full_probabilities.max(dim=-1)
        expected = torch.einsum(
            "b,bi,bij->bj",
            selected_probabilities,
            inputs,
            weight_bank[selected_indices],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor],
            )
            fit_trainer.fit(model, train_dataloaders=loader)

        torch.testing.assert_close(model.last_output, expected)
        logged = dict(fit_trainer.logged_metrics)
        torch.testing.assert_close(
            logged["parametric/router/weight_entropy"],
            torch.tensor(0.0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotIn("parametric/router/bias_entropy", logged)
        self.assertEqual(monitor._wrapped_methods, [])
        self.assertEqual(monitor._observations, {})
        self.assertEqual(monitor._utilization_histories, {})

    def test_real_dense_vector_lifecycle_preserves_the_expert_axis(self) -> None:
        layer = _dense_vector_parametric_layer()
        model = _ParametricTrainingModule(
            learning_rate=0.0,
            parametric_layer=layer,
        )
        monitor = ParametricLayerMonitorCallback(
            log_every_n_steps=1,
            history_size=2,
            log_per_slot_scalars=True,
        )
        observer = _UtilizationObserver(monitor)
        inputs = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]],
        )
        loader = DataLoader(TensorDataset(inputs), batch_size=3, shuffle=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="parametric-dense-vector",
            )
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor, observer],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=loader)
            tensorboard_logger.experiment.flush()
            events = EventAccumulator(
                tensorboard_logger.log_dir,
                size_guidance={"histograms": 0, "images": 0},
            )
            events.Reload()
            tags = events.Tags()

        torch.testing.assert_close(model.last_output, inputs)
        self.assertEqual(len(observer.weight_utilizations), 1)
        expected_utilization = torch.full((4,), 0.25)
        torch.testing.assert_close(
            observer.weight_utilizations[0],
            expected_utilization,
        )
        logged = dict(fit_trainer.logged_metrics)
        expected_scalars = {
            "parametric/router/weight_entropy": math.log(4.0),
            "parametric/mixture/weight_active_slots": 4.0,
            "parametric/mixture/weight_dead_slot_fraction": 0.0,
            "parametric/mixture/weight_max_utilization": 0.25,
            "parametric/mixture/weight_min_utilization": 0.25,
            "parametric/mixture/weight_slot_0_utilization": 0.25,
            "parametric/mixture/weight_slot_1_utilization": 0.25,
            "parametric/mixture/weight_slot_2_utilization": 0.25,
            "parametric/mixture/weight_slot_3_utilization": 0.25,
        }
        for metric_name, expected in expected_scalars.items():
            self.assert_logged_close(logged, metric_name, expected)
        self.assertIn(
            "parametric/mixture/histogram/weight_utilization",
            tags["histograms"],
        )
        self.assertIn(
            "parametric/mixture/heatmap/weight_utilization",
            tags["images"],
        )
        self.assertEqual(monitor._wrapped_methods, [])
        self.assertEqual(monitor._observations, {})
        self.assertEqual(monitor._utilization_histories, {})

    def test_real_weight_and_bias_slots_keep_distinct_monitoring_values(
        self,
    ) -> None:
        layer = _parametric_layer(top_k=1)
        with torch.no_grad():
            layer.weights_router.model[0].model.weight_params.copy_(
                torch.tensor([[2.0, -2.0], [2.0, -2.0]])
            )
            layer.bias_router.model[0].model.weight_params.copy_(
                torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
            )
            layer.weight_mixture_model.parameter_bank.copy_(
                torch.tensor(
                    [
                        [[1.5, 0.0], [0.0, 0.5]],
                        [[4.0, 4.0], [4.0, 4.0]],
                    ]
                )
            )
            layer.bias_mixture_model.parameter_bank.copy_(
                torch.tensor([[-4.0, -4.0], [3.0, 0.0]])
            )
        layer.weight_mixture_model.clip_range = 1.0
        layer.bias_mixture_model.clip_range = 2.0
        model = _ParametricTrainingModule(
            learning_rate=0.0,
            parametric_layer=layer,
        )
        monitor = ParametricLayerMonitorCallback(
            log_every_n_steps=1,
            log_per_slot_scalars=True,
        )
        inputs = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        loader = DataLoader(TensorDataset(inputs), batch_size=2, shuffle=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor],
            )
            fit_trainer.fit(model, train_dataloaders=loader)

        logged = dict(fit_trainer.logged_metrics)
        expected_scalars = {
            "parametric/parametric/weight_clip_saturation_fraction": 0.25,
            "parametric/parametric/bias_clip_saturation_fraction": 0.5,
            "parametric/router/weight_entropy": 0.0,
            "parametric/router/bias_entropy": 0.0,
            "parametric/mixture/weight_slot_0_utilization": 1.0,
            "parametric/mixture/weight_slot_1_utilization": 0.0,
            "parametric/mixture/bias_slot_0_utilization": 0.0,
            "parametric/mixture/bias_slot_1_utilization": 1.0,
        }
        for metric_name, expected in expected_scalars.items():
            self.assert_logged_close(logged, metric_name, expected)

    def test_real_rectangular_affine_omits_delta_metrics(self) -> None:
        affine_weights = torch.tensor([[1.0, -1.0, 2.0], [0.5, 3.0, -2.0]])
        layer = _weight_only_parametric_layer(
            input_dim=2,
            output_dim=3,
            top_k=2,
            num_experts=2,
            router_weights=torch.zeros(2, 2),
            weight_bank=torch.stack((affine_weights, affine_weights)),
        )
        model = _ParametricTrainingModule(
            learning_rate=0.0,
            parametric_layer=layer,
        )
        monitor = ParametricLayerMonitorCallback(log_every_n_steps=1)
        inputs = torch.tensor([[1.0, 2.0], [-2.0, 0.5]])
        loader = DataLoader(TensorDataset(inputs), batch_size=2, shuffle=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor],
            )
            fit_trainer.fit(model, train_dataloaders=loader)

        mixture_scale = 1.0 / (1.0 + 1e-6)
        torch.testing.assert_close(
            model.last_output,
            mixture_scale * (inputs @ affine_weights),
        )
        logged = dict(fit_trainer.logged_metrics)
        self.assertIn("parametric/parametric/affine/output_norm", logged)
        self.assertIn(
            "parametric/parametric/affine/relative_output_norm",
            logged,
        )
        self.assertNotIn("parametric/parametric/affine/delta_norm", logged)
        self.assertNotIn(
            "parametric/parametric/affine/relative_delta_norm",
            logged,
        )

    def test_real_zero_input_uses_exact_relative_norm_floor(self) -> None:
        model = _ParametricTrainingModule(learning_rate=0.0)
        monitor = ParametricLayerMonitorCallback(log_every_n_steps=1)
        inputs = torch.zeros(2, 2)
        loader = DataLoader(TensorDataset(inputs), batch_size=2, shuffle=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor],
            )
            fit_trainer.fit(model, train_dataloaders=loader)

        mixture_scale = 1.0 / (1.0 + 1e-6)
        expected_output = torch.tensor(
            [[2.0 * mixture_scale, 0.0], [2.0 * mixture_scale, 0.0]]
        )
        torch.testing.assert_close(model.last_output, expected_output)
        output_norm = expected_output.norm()
        logged = dict(fit_trainer.logged_metrics)
        self.assert_logged_close(
            logged,
            "parametric/parametric/affine/relative_output_norm",
            output_norm / 1e-6,
        )
        self.assert_logged_close(
            logged,
            "parametric/parametric/affine/relative_delta_norm",
            output_norm / 1e-6,
        )

    def test_real_trainer_exception_restores_wrappers_and_clears_state(self) -> None:
        model = _ParametricTrainingModule(
            fail_after_forward=True,
            learning_rate=0.0,
        )
        layer = model.parametric
        original_forward = layer.forward
        original_generate = layer._generate_parameters
        original_affine = layer._compute_affine_transformation_callback
        monitor = ParametricLayerMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = _trainer(Path(temporary_directory), [monitor])
            with self.assertRaisesRegex(
                RuntimeError,
                "^deliberate parametric lifecycle failure$",
            ):
                fit_trainer.fit(model, train_dataloaders=_loader())

        self.assertTrue(_same_bound_method(layer.forward, original_forward))
        self.assertTrue(
            _same_bound_method(layer._generate_parameters, original_generate)
        )
        self.assertTrue(
            _same_bound_method(
                layer._compute_affine_transformation_callback,
                original_affine,
            )
        )
        self.assertEqual(monitor._wrapped_methods, [])
        self.assertEqual(monitor._observations, {})
        self.assertEqual(monitor._utilization_histories, {})

    def test_real_trainer_rejects_duplicate_monitors_before_wrapping(
        self,
    ) -> None:
        model = _ParametricTrainingModule(learning_rate=0.0)
        layer = model.parametric
        original_forward = layer.forward
        original_generate = layer._generate_parameters
        original_affine = layer._compute_affine_transformation_callback
        monitors = (
            ParametricLayerMonitorCallback(log_every_n_steps=1),
            ParametricLayerMonitorCallback(log_every_n_steps=1),
        )
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        skip_mask = torch.ones(2, 1)
        expected = layer(inputs, skip_mask)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = _trainer(
                Path(temporary_directory),
                list(monitors),
            )
            with self.assertRaisesRegex(
                ValueError,
                "^Only one ParametricLayerMonitorCallback may be configured "
                "per Trainer\\.$",
            ):
                fit_trainer.fit(model, train_dataloaders=_loader())

        actual = layer(inputs, skip_mask)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            if actual_value is None or expected_value is None:
                self.assertIs(actual_value, expected_value)
            else:
                torch.testing.assert_close(actual_value, expected_value)
        self.assertTrue(_same_bound_method(layer.forward, original_forward))
        self.assertTrue(
            _same_bound_method(layer._generate_parameters, original_generate)
        )
        self.assertTrue(
            _same_bound_method(
                layer._compute_affine_transformation_callback,
                original_affine,
            )
        )
        for monitor in monitors:
            self.assertEqual(monitor._wrapped_methods, [])
            self.assertEqual(monitor._observations, {})
            self.assertEqual(monitor._utilization_histories, {})

    def test_caught_layer_failure_releases_observation_immediately(self) -> None:
        model = _CaughtLayerFailureTrainingModule()
        monitor = ParametricLayerMonitorCallback(log_every_n_steps=1)
        observer = _ObservationObserver(monitor)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = _trainer(
                Path(temporary_directory),
                [monitor, observer],
            )
            fit_trainer.fit(model, train_dataloaders=_loader())

        self.assertEqual(
            model.caught_errors,
            [
                "Input must be a 2D matrix (batch, input_dim), got 3D tensor "
                "with shape (1, 2, 2)."
            ],
        )
        self.assertEqual(observer.counts, [0])
        self.assertEqual(monitor._observations, {})


if __name__ == "__main__":
    unittest.main()
