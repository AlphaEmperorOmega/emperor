import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import torch

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsLayerState,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.experts._config import (
    _MixtureOfExpertsMapConfig,
    _MixtureOfExpertsReduceConfig,
)
from emperor.experts._layers.map import MixtureOfExpertsMap
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._layers.reduce import MixtureOfExpertsReduce
from emperor.experts._model import MixtureOfExpertsModel
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "MixtureOfExpertsConfig",
    "MixtureOfExpertsLayerConfig",
    "MixtureOfExpertsModelConfig",
    "DroppedTokenOptions",
    "ExpertWeightingPositionOptions",
    "RoutingInitializationMode",
    "MixtureOfExpertsLayerState",
)

EXPECTED_OWNERS = {
    "MixtureOfExpertsConfig": "emperor.experts._config",
    "MixtureOfExpertsLayerConfig": "emperor.experts._config",
    "MixtureOfExpertsModelConfig": "emperor.experts._config",
    "DroppedTokenOptions": "emperor.experts._options",
    "ExpertWeightingPositionOptions": "emperor.experts._options",
    "RoutingInitializationMode": "emperor.experts._options",
    "MixtureOfExpertsLayerState": "emperor.experts._state",
}

MIXTURE_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "top_k",
    "num_experts",
    "capacity_factor",
    "dropped_token_behavior",
    "compute_expert_mixture_flag",
    "weighted_parameters_flag",
    "weighting_position_option",
    "routing_initialization_mode",
    "sampler_config",
    "expert_model_config",
)

MODEL_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "top_k",
    "routing_initialization_mode",
    "sampler_config",
    "stack_config",
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

STATE_FIELDS = (
    "hidden",
    "loss",
    "halting_state",
    "probabilities",
    "indices",
    "skip_mask",
)

RNG_DIGESTS = {
    "disabled_sparse": (
        "ca7b41af6076101b3b70a6c067256e45dd5f4d13c843d2ac4c22a54c68427e84"
    ),
    "layer_sparse": (
        "ac0fcf874f878b385c00157c98f4f29fecbd8a80f478ae73b444a4dafa31db7f"
    ),
    "disabled_dense": (
        "ca7b41af6076101b3b70a6c067256e45dd5f4d13c843d2ac4c22a54c68427e84"
    ),
    "map": "ca7b41af6076101b3b70a6c067256e45dd5f4d13c843d2ac4c22a54c68427e84",
    "reduce": ("ca7b41af6076101b3b70a6c067256e45dd5f4d13c843d2ac4c22a54c68427e84"),
    "shared_model": (
        "b3feaf9e0624763bb3ba1df52f3e9f5c96f75adf1a70c2ae7927892e304417ee"
    ),
}

SEEDED_OUTPUT_DIGEST = (
    "4e217b4a21e10e7596678e57ec8645e0667a1372bbae0920d48692ba742f39c2"
)


def _linear_stack(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        ),
    )


def _sampler_config(top_k: int = 2) -> SamplerConfig:
    return SamplerConfig(
        top_k=top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        noisy_topk_flag=False,
        num_experts=3,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=RouterConfig(
            input_dim=4,
            num_experts=3,
            noisy_topk_flag=False,
            model_config=_linear_stack(4, 3),
        ),
    )


def _mixture_config(
    routing_mode: RoutingInitializationMode,
    *,
    top_k: int = 2,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=4,
        output_dim=4,
        top_k=top_k,
        num_experts=3,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
        routing_initialization_mode=routing_mode,
        sampler_config=_sampler_config(top_k),
        expert_model_config=_linear_stack(4, 4),
    )


def _disabled_sparse() -> MixtureOfExperts:
    return _mixture_config(RoutingInitializationMode.DISABLED).build()


def _layer_sparse() -> MixtureOfExperts:
    return _mixture_config(RoutingInitializationMode.LAYER).build()


def _disabled_dense() -> MixtureOfExperts:
    return _mixture_config(RoutingInitializationMode.DISABLED, top_k=3).build()


def _map() -> MixtureOfExpertsMap:
    config = _MixtureOfExpertsMapConfig(
        _mixture_config(RoutingInitializationMode.SHARED)
    )
    return config.build()


def _reduce() -> MixtureOfExpertsReduce:
    config = _MixtureOfExpertsReduceConfig(
        _mixture_config(RoutingInitializationMode.SHARED)
    )
    return config.build()


def _shared_model() -> MixtureOfExpertsModel:
    expert_layer_config = MixtureOfExpertsLayerConfig(
        activation=ActivationOptions.RELU,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=_mixture_config(RoutingInitializationMode.SHARED),
    )
    stack_config = LayerStackConfig(
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        num_layers=2,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=expert_layer_config,
    )
    config = MixtureOfExpertsModelConfig(
        input_dim=4,
        output_dim=4,
        top_k=2,
        routing_initialization_mode=RoutingInitializationMode.SHARED,
        sampler_config=_sampler_config(),
        stack_config=stack_config,
    )
    return config.build()


def _expert_keys(prefix: str = "") -> tuple[str, ...]:
    return tuple(
        f"{prefix}expert_modules.{expert_index}.layers.0.model.weight_params"
        for expert_index in range(3)
    )


SAMPLER_KEYS = (
    "sampler.sampler_model.default_loss",
    "sampler.sampler_model.auxiliary_loss_model.default_loss",
    "sampler.router.model.layers.0.model.weight_params",
)

SHARED_SAMPLER_KEYS = (
    "shared_sampler.sampler_model.default_loss",
    "shared_sampler.sampler_model.auxiliary_loss_model.default_loss",
    "shared_sampler.router.model.layers.0.model.weight_params",
)

SHARED_EXPERT_KEYS = tuple(
    key
    for layer_index in range(2)
    for key in _expert_keys(f"expert_stack.layers.{layer_index}.model.")
)


class TestExpertsPublicInterface(unittest.TestCase):
    def test_expert_forward_keeps_routing_arguments_and_appends_skip_mask(self):
        for model_type in (
            MixtureOfExperts,
            MixtureOfExpertsMap,
            MixtureOfExpertsReduce,
        ):
            with self.subTest(model_type=model_type.__name__):
                parameters = inspect.signature(model_type.forward).parameters
                self.assertEqual(
                    tuple(parameters),
                    ("self", "input_batch", "probabilities", "indices", "skip_mask"),
                )
                self.assertIsNone(parameters["probabilities"].default)
                self.assertIsNone(parameters["indices"].default)
                self.assertIsNone(parameters["skip_mask"].default)

    def test_exact_config_driven_exports_use_an_ordinary_interface(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys

import emperor.experts as experts

expected_eager_modules = (
    "emperor.experts._config",
    "emperor.experts._options",
    "emperor.experts._state",
)
heavy_modules = (
    "emperor.experts._model",
    "emperor.experts._validation.mixture",
    "emperor.experts._validation.model",
    "emperor.experts._routing.capacity",
    "emperor.experts._routing.weighting",
    "emperor.experts._layers.mixture",
    "emperor.experts._layers.layer",
    "emperor.experts._layers.map",
    "emperor.experts._layers.reduce",
)
runtime_loaded = {
    "lightning": "lightning" in sys.modules,
}
owners = {name: getattr(experts, name).__module__ for name in experts.__all__}

print(json.dumps({
    "all": experts.__all__,
    "expected_eager_modules": {
        name: name in sys.modules for name in expected_eager_modules
    },
    "heavy_modules": {name: name in sys.modules for name in heavy_modules},
    "owners": owners,
    "private_exports": {
        name: hasattr(experts, name)
        for name in (
            "ExpertCapacityHandler",
            "ExpertInputData",
            "ExpertWeightingHandler",
            "MixtureOfExperts",
            "MixtureOfExpertsLayer",
            "MixtureOfExpertsMap",
            "MixtureOfExpertsModel",
            "MixtureOfExpertsModelValidator",
            "MixtureOfExpertsReduce",
            "MixtureOfExpertsValidator",
        )
    },
    "runtime_loaded": runtime_loaded,
    "shortcut_attributes": {
        "__getattr__": hasattr(experts, "__getattr__"),
        "_LAZY_EXPORTS": hasattr(experts, "_LAZY_EXPORTS"),
    },
}))
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(
                    Path(tempfile.gettempdir()) / "matplotlib-experts-interface"
                ),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["all"]), EXPECTED_EXPORTS)
        self.assertEqual(result["owners"], EXPECTED_OWNERS)
        self.assertEqual(
            result["expected_eager_modules"],
            dict.fromkeys(result["expected_eager_modules"], True),
        )
        self.assertEqual(
            result["heavy_modules"],
            dict.fromkeys(result["heavy_modules"], False),
        )
        self.assertEqual(
            result["private_exports"],
            dict.fromkeys(result["private_exports"], False),
        )
        self.assertEqual(
            result["runtime_loaded"],
            {"lightning": False},
        )
        self.assertEqual(
            result["shortcut_attributes"],
            {"__getattr__": False, "_LAZY_EXPORTS": False},
        )

    def test_removed_concrete_implementation_imports_fail(self):
        for implementation_name in (
            "MixtureOfExperts",
            "MixtureOfExpertsMap",
            "MixtureOfExpertsModel",
            "MixtureOfExpertsReduce",
        ):
            with self.subTest(implementation=implementation_name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"from emperor.experts import {implementation_name}",
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ImportError", completed.stderr)

    def test_config_enum_and_state_schemas_are_preserved(self):
        self.assertEqual(
            tuple(field.name for field in fields(MixtureOfExpertsConfig)),
            MIXTURE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(MixtureOfExpertsModelConfig)),
            MODEL_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(MixtureOfExpertsLayerConfig)),
            LAYER_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(MixtureOfExpertsLayerState)),
            STATE_FIELDS,
        )
        self.assertEqual(
            tuple((option.name, option.value) for option in DroppedTokenOptions),
            (("ZEROS", 1), ("IDENTITY", 2)),
        )
        self.assertEqual(
            tuple(
                (option.name, option.value) for option in ExpertWeightingPositionOptions
            ),
            (("BEFORE_EXPERTS", 1), ("AFTER_EXPERTS", 2)),
        )
        self.assertEqual(
            tuple((option.name, option.value) for option in RoutingInitializationMode),
            (("DISABLED", 1), ("SHARED", 2), ("LAYER", 3)),
        )

    def test_exact_children_state_keys_and_strict_load_are_preserved(self):
        cases = (
            (
                "disabled_sparse",
                _disabled_sparse,
                ("expert_modules",),
                _expert_keys(),
            ),
            (
                "layer_sparse",
                _layer_sparse,
                ("sampler", "expert_modules"),
                (*SAMPLER_KEYS, *_expert_keys()),
            ),
            (
                "disabled_dense",
                _disabled_dense,
                ("expert_modules",),
                _expert_keys(),
            ),
            ("map", _map, ("expert_modules",), _expert_keys()),
            ("reduce", _reduce, ("expert_modules",), _expert_keys()),
            (
                "shared_model",
                _shared_model,
                ("shared_sampler", "expert_stack"),
                (*SHARED_SAMPLER_KEYS, *SHARED_EXPERT_KEYS),
            ),
        )

        for name, factory, expected_children, expected_keys in cases:
            with self.subTest(name=name):
                model = factory()
                self.assertEqual(tuple(model._modules), expected_children)
                self.assertEqual(tuple(model.state_dict()), expected_keys)

                restored = factory()
                load_result = restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(load_result.missing_keys, [])
                self.assertEqual(load_result.unexpected_keys, [])
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value)

    def test_seeded_construction_rng_contract_is_preserved(self):
        factories = {
            "disabled_sparse": _disabled_sparse,
            "layer_sparse": _layer_sparse,
            "disabled_dense": _disabled_dense,
            "map": _map,
            "reduce": _reduce,
            "shared_model": _shared_model,
        }

        for name, factory in factories.items():
            with self.subTest(name=name), torch.random.fork_rng():
                torch.manual_seed(7)
                factory()
                digest = hashlib.sha256(
                    torch.random.get_rng_state().numpy().tobytes()
                ).hexdigest()

                self.assertEqual(digest, RNG_DIGESTS[name])

    def test_seeded_sparse_output_contract_is_preserved(self):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.random.fork_rng():
                torch.manual_seed(7)
                model = _disabled_sparse().eval()
                inputs = torch.tensor(
                    (
                        (0.25, -0.5, 0.75, 1.0),
                        (-1.0, 0.5, 0.125, -0.25),
                        (0.3, 0.2, -0.1, 0.4),
                    )
                )
                probabilities = torch.tensor(((0.7, 0.3), (0.6, 0.4), (0.55, 0.45)))
                indices = torch.tensor(((0, 1), (2, 1), (1, 0)))

                with torch.no_grad():
                    output, skip_mask, loss = model(inputs, probabilities, indices)
        finally:
            torch.set_default_dtype(previous_dtype)

        self.assertIsNone(skip_mask)
        payload = (
            output.contiguous().numpy().tobytes() + loss.contiguous().numpy().tobytes()
        )
        self.assertEqual(hashlib.sha256(payload).hexdigest(), SEEDED_OUTPUT_DIGEST)

    def test_invalid_config_is_rejected_before_rng_consumption(self):
        config = _mixture_config(RoutingInitializationMode.DISABLED)
        config.input_dim = 0

        with torch.random.fork_rng():
            torch.manual_seed(17)
            expected_next_values = torch.randn(8)

            torch.manual_seed(17)
            with self.assertRaises(ValueError):
                config.build()
            actual_next_values = torch.randn(8)

        torch.testing.assert_close(actual_next_values, expected_next_values)


if __name__ == "__main__":
    unittest.main()
