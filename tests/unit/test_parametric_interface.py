import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from emperor.experts import RoutingInitializationMode
from emperor.parametric import (
    AdaptiveMixtureConfig,
    AdaptiveRouterOptions,
    ClipParameterOptions,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
    ParametricLayerMonitorCallback,
    ParametricLayerState,
    VectorRouterConfig,
    VectorWeightsMixtureConfig,
)
from unit.test_parametric import ParametricPresetMixin

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "AdaptiveRouterOptions",
    "ParametricLayerConfig",
    "ParametricLayerHandlerConfig",
    "ParametricLayer",
    "ParametricLayerMonitorCallback",
    "ParametricLayerHandler",
    "ParametricLayerState",
    "AdaptiveMixtureBase",
    "AdaptiveMixtureConfig",
    "ClipParameterOptions",
    "VectorWeightsMixtureConfig",
    "MatrixWeightsMixtureConfig",
    "MatrixBiasMixtureConfig",
    "GeneratorWeightsMixtureConfig",
    "GeneratorBiasMixtureConfig",
    "VectorWeightsMixture",
    "MatrixWeightsMixture",
    "MatrixBiasMixture",
    "GeneratorWeightsMixture",
    "GeneratorBiasMixture",
    "VectorRouterConfig",
    "VectorRouterModel",
)

EXPECTED_OWNERS = {
    "AdaptiveRouterOptions": "emperor.parametric._config",
    "ParametricLayerConfig": "emperor.parametric._config",
    "ParametricLayerHandlerConfig": "emperor.parametric._config",
    "ParametricLayer": "emperor.parametric._layer",
    "ParametricLayerMonitorCallback": "emperor.parametric._monitoring",
    "ParametricLayerHandler": "emperor.parametric._handlers",
    "ParametricLayerState": "emperor.parametric._state",
    "AdaptiveMixtureBase": "emperor.parametric._mixtures.base",
    "AdaptiveMixtureConfig": "emperor.parametric._mixtures.config",
    "ClipParameterOptions": "emperor.parametric._mixtures.config",
    "VectorWeightsMixtureConfig": "emperor.parametric._mixtures.config",
    "MatrixWeightsMixtureConfig": "emperor.parametric._mixtures.config",
    "MatrixBiasMixtureConfig": "emperor.parametric._mixtures.config",
    "GeneratorWeightsMixtureConfig": "emperor.parametric._mixtures.config",
    "GeneratorBiasMixtureConfig": "emperor.parametric._mixtures.config",
    "VectorWeightsMixture": "emperor.parametric._mixtures.vector",
    "MatrixWeightsMixture": "emperor.parametric._mixtures.matrix",
    "MatrixBiasMixture": "emperor.parametric._mixtures.matrix",
    "GeneratorWeightsMixture": "emperor.parametric._mixtures.generator",
    "GeneratorBiasMixture": "emperor.parametric._mixtures.generator",
    "VectorRouterConfig": "emperor.parametric._routing",
    "VectorRouterModel": "emperor.parametric._routing",
}

PARAMETRIC_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "weight_mixture_config",
    "bias_mixture_config",
    "routing_initialization_mode",
    "router_config",
    "sampler_config",
    "adaptive_augmentation_config",
)

LAYER_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "activation",
    "residual_config",
    "dropout_probability",
    "layer_norm_position",
    "gate_config",
    "halting_config",
    "memory_config",
    "layer_model_config",
)

MIXTURE_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "top_k",
    "num_experts",
    "weighted_parameters_flag",
    "clip_parameter_option",
    "clip_range",
)

ROUTER_CONFIG_FIELDS = (
    "input_dim",
    "num_experts",
    "noisy_topk_flag",
    "model_config",
)

STATE_FIELDS = (
    "hidden",
    "loss",
    "halting_state",
    "skip_mask",
)

FLOAT = torch.float32
LONG = torch.int64

VECTOR_TOPOLOGY = (
    ("weight_mixture_model.parameter_bank", (4, 4, 4), FLOAT),
    ("weight_mixture_model.select_range", (1, 4, 1), LONG),
    ("weights_router.parameter_bank", (4, 4, 4), FLOAT),
    ("weights_router.model.layers.0.model.weight_params", (4, 4), FLOAT),
    ("sampler.sampler_model.default_loss", (), FLOAT),
    ("sampler.sampler_model.auxiliary_loss_model.default_loss", (), FLOAT),
)

MATRIX_SHARED_TOPOLOGY = (
    ("weight_mixture_model.parameter_bank", (4, 4, 3), FLOAT),
    ("bias_mixture_model.parameter_bank", (4, 3), FLOAT),
    ("weights_router.model.layers.0.model.weight_params", (4, 4), FLOAT),
    ("sampler.sampler_model.default_loss", (), FLOAT),
    ("sampler.sampler_model.auxiliary_loss_model.default_loss", (), FLOAT),
)

MATRIX_INDEPENDENT_TOPOLOGY = (
    ("weight_mixture_model.parameter_bank", (4, 4, 3), FLOAT),
    ("bias_mixture_model.parameter_bank", (4, 3), FLOAT),
    ("weights_router.model.layers.0.model.weight_params", (4, 4), FLOAT),
    ("bias_router.model.layers.0.model.weight_params", (4, 4), FLOAT),
    ("sampler.sampler_model.default_loss", (), FLOAT),
    ("sampler.sampler_model.auxiliary_loss_model.default_loss", (), FLOAT),
)

GENERATOR_INDEPENDENT_TOPOLOGY = (
    (
        "weight_mixture_model.input_vector_generator."
        "sampler.sampler_model.default_loss",
        (),
        FLOAT,
    ),
    (
        "weight_mixture_model.input_vector_generator."
        "sampler.sampler_model.auxiliary_loss_model.default_loss",
        (),
        FLOAT,
    ),
    (
        "weight_mixture_model.input_vector_generator."
        "sampler.router.model.layers.0.model.weight_params",
        (4, 4),
        FLOAT,
    ),
    *tuple(
        (
            "weight_mixture_model.input_vector_generator."
            f"expert_modules.{expert}.layers.0.model.weight_params",
            (4, 4),
            FLOAT,
        )
        for expert in range(4)
    ),
    (
        "weight_mixture_model.output_vector_generator."
        "sampler.sampler_model.default_loss",
        (),
        FLOAT,
    ),
    (
        "weight_mixture_model.output_vector_generator."
        "sampler.sampler_model.auxiliary_loss_model.default_loss",
        (),
        FLOAT,
    ),
    (
        "weight_mixture_model.output_vector_generator."
        "sampler.router.model.layers.0.model.weight_params",
        (4, 4),
        FLOAT,
    ),
    *tuple(
        (
            "weight_mixture_model.output_vector_generator."
            f"expert_modules.{expert}.layers.0.model.weight_params",
            (4, 3),
            FLOAT,
        )
        for expert in range(4)
    ),
    ("bias_mixture_model.bias_generator.sampler.sampler_model.default_loss", (), FLOAT),
    (
        "bias_mixture_model.bias_generator."
        "sampler.sampler_model.auxiliary_loss_model.default_loss",
        (),
        FLOAT,
    ),
    (
        "bias_mixture_model.bias_generator."
        "sampler.router.model.layers.0.model.weight_params",
        (4, 4),
        FLOAT,
    ),
    *tuple(
        (
            "bias_mixture_model.bias_generator."
            f"expert_modules.{expert}.layers.0.model.weight_params",
            (4, 3),
            FLOAT,
        )
        for expert in range(4)
    ),
)

GENERATOR_SHARED_TOPOLOGY = (
    *tuple(
        (
            "weight_mixture_model.input_vector_generator."
            f"expert_modules.{expert}.layers.0.model.weight_params",
            (4, 4),
            FLOAT,
        )
        for expert in range(4)
    ),
    *tuple(
        (
            "weight_mixture_model.output_vector_generator."
            f"expert_modules.{expert}.layers.0.model.weight_params",
            (4, 3),
            FLOAT,
        )
        for expert in range(4)
    ),
    *tuple(
        (
            "bias_mixture_model.bias_generator."
            f"expert_modules.{expert}.layers.0.model.weight_params",
            (4, 3),
            FLOAT,
        )
        for expert in range(4)
    ),
    ("weights_router.model.layers.0.model.weight_params", (4, 4), FLOAT),
    ("sampler.sampler_model.default_loss", (), FLOAT),
    ("sampler.sampler_model.auxiliary_loss_model.default_loss", (), FLOAT),
)

PRESETS = ParametricPresetMixin()


def _vector_config() -> ParametricLayerConfig:
    return PRESETS.parametric_config(
        weight_mixture_config=PRESETS.vector_weights_config(),
        routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        input_dim=4,
        output_dim=4,
    )


def _matrix_shared_config() -> ParametricLayerConfig:
    return PRESETS.parametric_config(
        weight_mixture_config=PRESETS.matrix_weights_config(),
        bias_mixture_config=PRESETS.matrix_bias_config(),
        routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
    )


def _matrix_independent_config() -> ParametricLayerConfig:
    return PRESETS.parametric_config(
        weight_mixture_config=PRESETS.matrix_weights_config(),
        bias_mixture_config=PRESETS.matrix_bias_config(),
        routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
    )


def _generator_independent_config() -> ParametricLayerConfig:
    return PRESETS.parametric_config(
        weight_mixture_config=PRESETS.generator_weights_config(),
        bias_mixture_config=PRESETS.generator_bias_config(),
        routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
    )


def _generator_shared_config() -> ParametricLayerConfig:
    return PRESETS.parametric_config(
        weight_mixture_config=PRESETS.generator_weights_config(
            RoutingInitializationMode.DISABLED
        ),
        bias_mixture_config=PRESETS.generator_bias_config(
            RoutingInitializationMode.DISABLED
        ),
        routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
    )


@dataclass(frozen=True)
class ParametricFixture:
    name: str
    config_factory: Callable[[], ParametricLayerConfig]
    children: tuple[str, ...]
    topology: tuple[tuple[str, tuple[int, ...], torch.dtype], ...]
    rng_digest: str
    output_digest: str


FIXTURES = (
    ParametricFixture(
        "vector_independent",
        _vector_config,
        (
            "adaptive_augmentation_model",
            "weight_mixture_model",
            "parameter_handler",
            "weights_router",
            "sampler",
        ),
        VECTOR_TOPOLOGY,
        "32b5e631a61674134ea375e2506ef151174bfbcb88c58bb04d6b4b66278a6f6a",
        "4ffd36b8beb25f34451fdad6f5a1c3c674dd416051005eaba745f43bcc9e3600",
    ),
    ParametricFixture(
        "matrix_shared_bias",
        _matrix_shared_config,
        (
            "adaptive_augmentation_model",
            "weight_mixture_model",
            "bias_mixture_model",
            "parameter_handler",
            "weights_router",
            "sampler",
        ),
        MATRIX_SHARED_TOPOLOGY,
        "d7086c34f9c826ea282b142a26e3c400bd7810b0db01e83a3fa02fd975c49eda",
        "68b42d6b18e9d6719c6ec2e6fc4145b269ca178ed8235ce0338c5faf28ce8e6b",
    ),
    ParametricFixture(
        "matrix_independent_bias",
        _matrix_independent_config,
        (
            "adaptive_augmentation_model",
            "weight_mixture_model",
            "bias_mixture_model",
            "parameter_handler",
            "weights_router",
            "bias_router",
            "sampler",
        ),
        MATRIX_INDEPENDENT_TOPOLOGY,
        "4f93baa408d1a4751cd29f8ab91a8f8939024ff32f4a6cbf121e90896aeda666",
        "98604a7e67a6ae9612b3d647d1d6a6247ed7ecf9aba2f16339a081180cc436ab",
    ),
    ParametricFixture(
        "generator_independent_bias",
        _generator_independent_config,
        (
            "adaptive_augmentation_model",
            "weight_mixture_model",
            "bias_mixture_model",
            "parameter_handler",
        ),
        GENERATOR_INDEPENDENT_TOPOLOGY,
        "41b37524319dd96edc255374e2aeb62f88d21cd2201fd6fa9a8cb3c44866b80a",
        "ccb4a0daf84bf819e6301fc5c98139f1df35913e88e1b3f086faf18a795ece41",
    ),
    ParametricFixture(
        "generator_shared_bias",
        _generator_shared_config,
        (
            "adaptive_augmentation_model",
            "weight_mixture_model",
            "bias_mixture_model",
            "parameter_handler",
            "weights_router",
            "sampler",
        ),
        GENERATOR_SHARED_TOPOLOGY,
        "eb4926538ae0a7674e0ad7324345fdda1b07474ed2b6d762f1ced6bef89c4d2c",
        "9421b18c7184c1efa0e8a31e66adaec697bfc1988510121d3a1bbd0eeb563cf1",
    ),
)


def _input(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        (
            (0.25, -0.50, 0.75, 1.00),
            (-1.00, 0.50, 0.125, -0.25),
        ),
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def _digest(*tensors: torch.Tensor | None) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TestParametricPublicInterface(unittest.TestCase):
    def test_exact_exports_resolve_eagerly_from_their_owning_modules(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys

import emperor.parametric as parametric

private_modules = (
    "emperor.parametric._config",
    "emperor.parametric._state",
    "emperor.parametric._layer",
    "emperor.parametric._handlers",
    "emperor.parametric._routing",
    "emperor.parametric._monitoring",
    "emperor.parametric._validation",
    "emperor.parametric._mixtures",
    "emperor.parametric._mixtures.base",
    "emperor.parametric._mixtures.config",
    "emperor.parametric._mixtures.validation",
    "emperor.parametric._mixtures.vector",
    "emperor.parametric._mixtures.matrix",
    "emperor.parametric._mixtures.generator",
)
before = {name: name in sys.modules for name in private_modules}
runtime_before = {
    "emperor.experts": "emperor.experts" in sys.modules,
    "lightning": "lightning" in sys.modules,
    "torch": "torch" in sys.modules,
}

import torch

torch.manual_seed(73)
expected_next_values = torch.randn(8)
torch.manual_seed(73)
owners = {name: getattr(parametric, name).__module__ for name in parametric.__all__}
actual_next_values = torch.randn(8)

print(json.dumps({
    "all": parametric.__all__,
    "before": before,
    "owners": owners,
    "private_exports": {
        name: hasattr(parametric, name)
        for name in (
            "AdaptiveMixtureValidator",
            "GeneratorMixtureBase",
            "GeneratorParameterHandler",
            "MatrixMixtureBase",
            "ParameterHandlerBase",
            "ParametricHandlerValidator",
            "ParametricLayerValidator",
            "VectorMixtureBase",
            "_ParametricObservation",
        )
    },
    "rng_unchanged": torch.equal(expected_next_values, actual_next_values),
    "runtime_before": runtime_before,
}))
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(
                    Path(tempfile.gettempdir()) / "matplotlib-parametric-interface"
                ),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["all"]), EXPECTED_EXPORTS)
        self.assertEqual(result["owners"], EXPECTED_OWNERS)
        expected_loaded = dict.fromkeys(result["before"], True)
        self.assertEqual(result["before"], expected_loaded)
        self.assertEqual(
            result["private_exports"],
            dict.fromkeys(result["private_exports"], False),
        )
        self.assertTrue(result["rng_unchanged"])
        self.assertEqual(
            result["runtime_before"],
            {"emperor.experts": True, "lightning": True, "torch": True},
        )

    def test_config_state_option_and_callback_contracts_are_preserved(self):
        self.assertEqual(
            tuple(field.name for field in fields(ParametricLayerConfig)),
            PARAMETRIC_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(ParametricLayerHandlerConfig)),
            LAYER_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(AdaptiveMixtureConfig)),
            MIXTURE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(VectorWeightsMixtureConfig)),
            MIXTURE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(MatrixWeightsMixtureConfig)),
            MIXTURE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(MatrixBiasMixtureConfig)),
            MIXTURE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(GeneratorWeightsMixtureConfig)),
            (*MIXTURE_CONFIG_FIELDS, "generator_config"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(GeneratorBiasMixtureConfig)),
            (*MIXTURE_CONFIG_FIELDS, "generator_config"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(VectorRouterConfig)),
            ROUTER_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(ParametricLayerState)),
            STATE_FIELDS,
        )

        for config_type in (
            ParametricLayerConfig,
            ParametricLayerHandlerConfig,
            AdaptiveMixtureConfig,
            VectorWeightsMixtureConfig,
            MatrixWeightsMixtureConfig,
            MatrixBiasMixtureConfig,
            GeneratorWeightsMixtureConfig,
            GeneratorBiasMixtureConfig,
            VectorRouterConfig,
        ):
            with self.subTest(config=config_type.__name__):
                config = config_type()
                self.assertTrue(
                    all(getattr(config, field.name) is None for field in fields(config))
                )

        self.assertEqual(
            tuple((option.name, option.value) for option in AdaptiveRouterOptions),
            (("SHARED_ROUTER", 1), ("INDEPENDENT_ROUTER", 2)),
        )
        self.assertEqual(
            tuple((option.name, option.value) for option in ClipParameterOptions),
            (("DISABLED", 0), ("BEFORE", 1), ("AFTER", 2)),
        )

        parameters = inspect.signature(ParametricLayerMonitorCallback).parameters
        self.assertEqual(
            tuple(parameters),
            ("log_every_n_steps", "history_size", "log_per_slot_scalars"),
        )
        self.assertEqual(parameters["log_every_n_steps"].default, 100)
        self.assertEqual(parameters["history_size"].default, 128)
        self.assertIs(parameters["log_per_slot_scalars"].default, False)

    def test_exact_children_state_topology_and_strict_loading_are_preserved(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(20260716)
                model = ParametricLayer(fixture.config_factory())
                self.assertEqual(tuple(model._modules), fixture.children)
                self.assertEqual(
                    tuple(
                        (name, tuple(value.shape), value.dtype)
                        for name, value in model.state_dict().items()
                    ),
                    fixture.topology,
                )

                torch.manual_seed(20260716)
                restored = ParametricLayer(fixture.config_factory())
                result = restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value)

    def test_seeded_construction_rng_and_forward_fingerprints_are_preserved(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(20260716)
                model = ParametricLayer(fixture.config_factory()).eval()
                self.assertEqual(
                    _digest(torch.random.get_rng_state()),
                    fixture.rng_digest,
                )

                output, skip_mask, loss = model(_input())

                self.assertIsNone(skip_mask)
                self.assertEqual(loss.shape, torch.Size([]))
                self.assertEqual(
                    _digest(output, skip_mask, loss),
                    fixture.output_digest,
                )

    def test_invalid_configs_are_rejected_before_rng_consumption(self):
        for fixture in FIXTURES:
            config = fixture.config_factory()
            config.input_dim = 0
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(17)
                expected_next_values = torch.randn(8)

                torch.manual_seed(17)
                with self.assertRaises(ValueError):
                    config.build()
                actual_next_values = torch.randn(8)

                torch.testing.assert_close(actual_next_values, expected_next_values)

    def test_seeded_payloads_preserve_variant_gradient_contracts(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(20260716)
                model = ParametricLayer(fixture.config_factory()).eval()
                input_batch = _input(requires_grad=True)
                output, _skip_mask, loss = model(input_batch)
                (output.square().sum() + loss).backward()

                self.assertIsNotNone(input_batch.grad)
                self.assertTrue(torch.isfinite(input_batch.grad).all())
                self.assertGreater(torch.count_nonzero(input_batch.grad).item(), 0)
                self._assert_variant_gradients(fixture.name, model)

    def _assert_variant_gradients(
        self,
        fixture_name: str,
        model: ParametricLayer,
    ) -> None:
        gradients = {
            name: parameter.grad for name, parameter in model.named_parameters()
        }

        def assert_nonzero(name: str) -> None:
            gradient = gradients[name]
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(torch.count_nonzero(gradient).item(), 0, name)

        if fixture_name == "vector_independent":
            assert_nonzero("weight_mixture_model.parameter_bank")
            assert_nonzero("weights_router.parameter_bank")
            self.assertIsNone(
                gradients["weights_router.model.layers.0.model.weight_params"]
            )
            return

        if fixture_name.startswith("matrix_"):
            assert_nonzero("weight_mixture_model.parameter_bank")
            assert_nonzero("bias_mixture_model.parameter_bank")
            assert_nonzero("weights_router.model.layers.0.model.weight_params")
            if fixture_name == "matrix_independent_bias":
                assert_nonzero("bias_router.model.layers.0.model.weight_params")
            return

        expert_gradients = [
            gradient
            for name, gradient in gradients.items()
            if ".expert_modules." in name
        ]
        self.assertTrue(any(gradient is None for gradient in expert_gradients))
        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient)
                for gradient in expert_gradients
            )
        )

        if fixture_name == "generator_independent_bias":
            self.assertIsNone(
                gradients[
                    "weight_mixture_model.input_vector_generator."
                    "sampler.router.model.layers.0.model.weight_params"
                ]
            )
            self.assertIsNone(
                gradients[
                    "weight_mixture_model.output_vector_generator."
                    "sampler.router.model.layers.0.model.weight_params"
                ]
            )
            assert_nonzero(
                "bias_mixture_model.bias_generator."
                "sampler.router.model.layers.0.model.weight_params"
            )
            return

        assert_nonzero("weights_router.model.layers.0.model.weight_params")


if __name__ == "__main__":
    unittest.main()
