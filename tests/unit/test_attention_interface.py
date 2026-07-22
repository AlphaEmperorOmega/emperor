import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from emperor.attention import (
    AttentionLayerState,
    IndependentAttentionConfig,
    MixerAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    MultiHeadAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.attention._runtime import QKV, AttentionMasks, AttentionRuntimeLayout
from emperor.experts import RoutingInitializationMode
from support.attention import build_attention_config

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "MultiHeadAttentionConfig",
    "SelfAttentionConfig",
    "SelfAttentionProjectionStrategy",
    "IndependentAttentionConfig",
    "MixtureOfAttentionHeadsConfig",
    "MixerAttentionConfig",
    "AttentionLayerState",
    "AttentionMonitorCallback",
)

EXPECTED_OWNERS = {
    "MultiHeadAttentionConfig": "emperor.attention._config",
    "SelfAttentionConfig": "emperor.attention._variants.self_attention.config",
    "SelfAttentionProjectionStrategy": (
        "emperor.attention._variants.self_attention.config"
    ),
    "IndependentAttentionConfig": ("emperor.attention._variants.independent.config"),
    "MixtureOfAttentionHeadsConfig": "emperor.attention._variants.mixture.config",
    "MixerAttentionConfig": "emperor.attention._variants.mixer.config",
    "AttentionLayerState": "emperor.attention._state",
    "AttentionMonitorCallback": "emperor.attention._monitoring.callback",
}

BASE_CONFIG_FIELDS = (
    "batch_size",
    "num_heads",
    "embedding_dim",
    "query_key_projection_dim",
    "value_projection_dim",
    "target_sequence_length",
    "source_sequence_length",
    "target_dtype",
    "dropout_probability",
    "zero_attention_flag",
    "causal_attention_mask_flag",
    "add_key_value_bias_flag",
    "average_attention_weights_flag",
    "return_attention_weights_flag",
    "batch_first_flag",
    "projection_model_config",
    "relative_positional_embedding_config",
)


@dataclass(frozen=True)
class AttentionFixture:
    name: str
    config_class: type[MultiHeadAttentionConfig]
    expected_output: tuple[float, ...]
    projection_strategy: SelfAttentionProjectionStrategy | None = None
    use_kv_expert_models_flag: bool = False


FIXTURES = (
    AttentionFixture(
        "self_fused",
        SelfAttentionConfig,
        (
            0.1038186027122618,
            0.7812755541254880,
            0.1286831165777521,
            0.8386745460119083,
        ),
        SelfAttentionProjectionStrategy.FUSED,
    ),
    AttentionFixture(
        "self_separate",
        SelfAttentionConfig,
        (
            0.3855337074709573,
            1.0117078612153028,
            0.4152020751266840,
            1.0751434706942185,
        ),
        SelfAttentionProjectionStrategy.SEPARATE,
    ),
    AttentionFixture(
        "independent",
        IndependentAttentionConfig,
        (
            -0.2355060046390766,
            -0.3757721916562812,
            -0.2411684159851785,
            -0.3736643920167895,
        ),
    ),
    AttentionFixture(
        "mixture_shared_kv",
        MixtureOfAttentionHeadsConfig,
        (
            -0.5288931771536676,
            -0.4772958568800849,
            -0.6131617677683937,
            -0.5457771599890396,
        ),
    ),
    AttentionFixture(
        "mixture_expert_kv",
        MixtureOfAttentionHeadsConfig,
        (
            0.1605987075211371,
            -0.3674997315156712,
            -0.0875965783555378,
            -0.5118885990634013,
        ),
        use_kv_expert_models_flag=True,
    ),
)

# Filled from the canonical seed-1501 fixture after construction. These values
# deliberately lock parameter-construction order independently of forward math.
RNG_DIGESTS = {
    "self_fused": ("80fe50672d97557505455cb570abc703a926cb3c7515bd9ae45eda4f06af22f2"),
    "self_separate": (
        "80fe50672d97557505455cb570abc703a926cb3c7515bd9ae45eda4f06af22f2"
    ),
    "independent": ("80fe50672d97557505455cb570abc703a926cb3c7515bd9ae45eda4f06af22f2"),
    "mixture_shared_kv": (
        "01f50975bcba240534ff984c4c809d60abf72d861016a7276843024b25498b0c"
    ),
    "mixture_expert_kv": (
        "47c8793a0f054e70e667b72c744f6a41c4a8220874f00a7dc6d7a97fa0e34b35"
    ),
}

_BIAS_TOPOLOGY = (
    ("bias.key_bias_vector", (1, 1, 2), torch.float64),
    ("bias.value_bias_vector", (1, 1, 2), torch.float64),
)
_OUTPUT_TOPOLOGY = (
    (
        "projector.output_model.layers.0.model.weight_params",
        (2, 2),
        torch.float64,
    ),
    ("projector.output_model.layers.0.model.bias_params", (2,), torch.float64),
)
_SEPARATE_PROJECTION_TOPOLOGY = tuple(
    entry
    for projection in ("query", "key", "value")
    for entry in (
        (
            f"projector.{projection}_model.layers.0.model.weight_params",
            (2, 2),
            torch.float64,
        ),
        (
            f"projector.{projection}_model.layers.0.model.bias_params",
            (2,),
            torch.float64,
        ),
    )
)


def _expert_projection_topology(projection: str) -> tuple:
    return tuple(
        entry
        for expert_index in range(2)
        for entry in (
            (
                "projector."
                f"{projection}_model.expert_modules.{expert_index}."
                "layers.0.model.weight_params",
                (2, 2),
                torch.float64,
            ),
            (
                "projector."
                f"{projection}_model.expert_modules.{expert_index}."
                "layers.0.model.bias_params",
                (2,),
                torch.float64,
            ),
        )
    )


_MIXTURE_SAMPLER_TOPOLOGY = (
    ("projector.sampler.sampler_model.default_loss", (), torch.float64),
    (
        "projector.sampler.sampler_model.auxiliary_loss_model.default_loss",
        (),
        torch.float64,
    ),
    (
        "projector.sampler.router.model.layers.0.model.weight_params",
        (2, 2),
        torch.float64,
    ),
    (
        "projector.sampler.router.model.layers.1.model.weight_params",
        (2, 2),
        torch.float64,
    ),
)

EXPECTED_STATE_TOPOLOGY = {
    "self_fused": (
        *_BIAS_TOPOLOGY,
        *_OUTPUT_TOPOLOGY,
        (
            "projector.qkv_model.layers.0.model.weight_params",
            (2, 6),
            torch.float64,
        ),
        ("projector.qkv_model.layers.0.model.bias_params", (6,), torch.float64),
    ),
    "self_separate": (
        *_BIAS_TOPOLOGY,
        *_OUTPUT_TOPOLOGY,
        *_SEPARATE_PROJECTION_TOPOLOGY,
    ),
    "independent": (
        *_BIAS_TOPOLOGY,
        *_OUTPUT_TOPOLOGY,
        *_SEPARATE_PROJECTION_TOPOLOGY,
    ),
    "mixture_shared_kv": (
        *_BIAS_TOPOLOGY,
        *_expert_projection_topology("output"),
        *_expert_projection_topology("query"),
        (
            "projector.key_model.layers.0.model.weight_params",
            (2, 2),
            torch.float64,
        ),
        ("projector.key_model.layers.0.model.bias_params", (2,), torch.float64),
        (
            "projector.value_model.layers.0.model.weight_params",
            (2, 2),
            torch.float64,
        ),
        ("projector.value_model.layers.0.model.bias_params", (2,), torch.float64),
        *_MIXTURE_SAMPLER_TOPOLOGY,
    ),
    "mixture_expert_kv": (
        *_BIAS_TOPOLOGY,
        *_expert_projection_topology("output"),
        *_expert_projection_topology("query"),
        *_expert_projection_topology("key"),
        *_expert_projection_topology("value"),
        *_MIXTURE_SAMPLER_TOPOLOGY,
    ),
}


def _fixture_config(fixture: AttentionFixture) -> MultiHeadAttentionConfig:
    kwargs = dict(
        config_class=fixture.config_class,
        batch_size=1,
        num_heads=1,
        embedding_dim=2,
        query_key_projection_dim=2,
        value_projection_dim=2,
        target_sequence_length=2,
        source_sequence_length=2,
        dropout_probability=0.0,
        add_key_value_bias_flag=True,
        return_attention_weights_flag=False,
    )
    if fixture.config_class is SelfAttentionConfig:
        kwargs["self_attention_projection_strategy"] = fixture.projection_strategy
    if fixture.config_class is MixtureOfAttentionHeadsConfig:
        kwargs.update(
            use_kv_expert_models_flag=fixture.use_kv_expert_models_flag,
            experts_top_k=1,
            experts_num_experts=2,
            experts_routing_initialization_mode=RoutingInitializationMode.LAYER,
            experts_stack_num_layers=1,
        )
    return build_attention_config(**kwargs)


def _fixture_model(fixture: AttentionFixture):
    return _fixture_config(fixture).build().to(dtype=torch.float64).eval()


def _fixture_inputs(fixture: AttentionFixture):
    generator = torch.Generator(device="cpu").manual_seed(1502)
    query = torch.randn(
        2,
        1,
        2,
        generator=generator,
        dtype=torch.float64,
    ).requires_grad_(True)
    key = torch.randn(2, 1, 2, generator=generator, dtype=torch.float64)
    value = torch.randn(2, 1, 2, generator=generator, dtype=torch.float64)
    if fixture.config_class is SelfAttentionConfig or (
        fixture.config_class is MixtureOfAttentionHeadsConfig
        and fixture.use_kv_expert_models_flag
    ):
        return query, query, query
    return query, key.requires_grad_(True), value.requires_grad_(True)


def _clone_inputs_preserving_aliases(
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    clones: dict[int, torch.Tensor] = {}
    result = []
    for value in inputs:
        clone = clones.get(id(value))
        if clone is None:
            clone = value.detach().clone().requires_grad_(True)
            clones[id(value)] = clone
        result.append(clone)
    return tuple(result)


class TestAttentionPublicInterface(unittest.TestCase):
    def test_root_interface_eagerly_exports_public_records_and_callbacks(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys

import emperor.attention as attention

expected_eager_modules = (
    "emperor.attention._config",
    "emperor.attention._state",
    "emperor.attention._variants.independent.config",
    "emperor.attention._variants.self_attention.config",
    "emperor.attention._variants.mixture.config",
    "emperor.attention._variants.mixer.config",
)
heavy_modules = (
    "emperor.attention._base",
    "emperor.attention._runtime",
    "emperor.attention._validation",
    "emperor.attention._monitoring.callback",
    "emperor.attention._monitoring.diagnostics",
    "emperor.attention._ops.batching",
    "emperor.attention._ops.bias",
    "emperor.attention._ops.masking",
    "emperor.attention._ops.processing",
    "emperor.attention._ops.projection",
    "emperor.attention._ops.reshaping",
    "emperor.attention._ops.zero_attention",
    "emperor.attention._variants.independent.layer",
    "emperor.attention._variants.independent.processing",
    "emperor.attention._variants.independent.projection",
    "emperor.attention._variants.independent.validation",
    "emperor.attention._variants.self_attention.layer",
    "emperor.attention._variants.self_attention.processing",
    "emperor.attention._variants.self_attention.projection",
    "emperor.attention._variants.self_attention.validation",
    "emperor.attention._variants.mixture.layer",
    "emperor.attention._variants.mixture.bias",
    "emperor.attention._variants.mixture.masking",
    "emperor.attention._variants.mixture.monitoring",
    "emperor.attention._variants.mixture.processing",
    "emperor.attention._variants.mixture.projection",
    "emperor.attention._variants.mixture.reshaping",
    "emperor.attention._variants.mixture.validation",
    "emperor.attention._variants.mixture.zero_attention",
    "emperor.attention._variants.mixer.layer",
    "emperor.attention._variants.mixer.validation",
)
runtime_loaded = {
    "emperor.experts": "emperor.experts" in sys.modules,
    "lightning": "lightning" in sys.modules,
    "torch": "torch" in sys.modules,
}
owners = {name: getattr(attention, name).__module__ for name in attention.__all__}

print(json.dumps({
    "all": attention.__all__,
    "expected_eager_modules": {
        name: name in sys.modules for name in expected_eager_modules
    },
    "heavy_modules": {name: name in sys.modules for name in heavy_modules},
    "owners": owners,
    "private_exports": {
        name: hasattr(attention, name)
        for name in (
            "AttentionMasks",
            "AttentionMonitorCallback",
            "AttentionRuntimeLayout",
            "AttentionValidatorBase",
            "IndependentAttention",
            "IndependentProcessor",
            "MixtureOfAttentionHeads",
            "MixerAttention",
            "MultiHeadAttentionAbstract",
            "MultiHeadAttentionValidator",
            "QKV",
            "SelfAttention",
        )
    },
    "runtime_loaded": runtime_loaded,
    "shortcut_attributes": {
        "__getattr__": hasattr(attention, "__getattr__"),
        "_LAZY_EXPORTS": hasattr(attention, "_LAZY_EXPORTS"),
    },
}))
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(
                    Path(tempfile.gettempdir()) / "matplotlib-attention-interface"
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
        self.assertTrue(result["heavy_modules"]["emperor.attention._runtime"])
        self.assertTrue(
            result["heavy_modules"]["emperor.attention._monitoring.callback"]
        )
        expected_private_exports = dict.fromkeys(result["private_exports"], False)
        expected_private_exports["AttentionMonitorCallback"] = True
        self.assertEqual(result["private_exports"], expected_private_exports)
        self.assertEqual(
            result["runtime_loaded"],
            {"emperor.experts": False, "lightning": True, "torch": True},
        )
        self.assertEqual(
            result["shortcut_attributes"],
            {"__getattr__": False, "_LAZY_EXPORTS": False},
        )

    def test_monitoring_has_its_own_explicit_interface(self):
        attention = __import__("emperor.attention", fromlist=["attention"])
        monitoring = __import__(
            "emperor.attention.monitoring",
            fromlist=["monitoring"],
        )

        self.assertEqual(monitoring.__all__, ("AttentionMonitorCallback",))
        self.assertEqual(
            monitoring.AttentionMonitorCallback.__module__,
            "emperor.attention._monitoring.callback",
        )
        self.assertIs(
            attention.AttentionMonitorCallback,
            monitoring.AttentionMonitorCallback,
        )

    def test_config_state_runtime_and_enum_schemas_are_preserved(self):
        self.assertEqual(
            tuple(field.name for field in fields(MultiHeadAttentionConfig)),
            BASE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(IndependentAttentionConfig)),
            BASE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(SelfAttentionConfig)),
            (*BASE_CONFIG_FIELDS, "projection_strategy"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(MixtureOfAttentionHeadsConfig)),
            (*BASE_CONFIG_FIELDS, "experts_config", "use_kv_expert_models_flag"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(MixerAttentionConfig)),
            (
                "embedding_dim",
                "sequence_length",
                "batch_first_flag",
                "causal_attention_mask_flag",
                "mixing_model_config",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(AttentionLayerState)),
            ("hidden", "loss", "halting_state", "key_padding_mask", "attention_mask"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(QKV)),
            ("query", "key", "value"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(AttentionMasks)),
            ("key_padding_mask", "attention_mask"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(AttentionRuntimeLayout)),
            (
                "batch_size",
                "target_sequence_length",
                "source_sequence_length",
                "input_was_batched",
                "input_was_batch_first",
                "source_extension_count",
            ),
        )
        self.assertEqual(
            tuple(
                (option.name, option.value)
                for option in SelfAttentionProjectionStrategy
            ),
            (("FUSED", 0), ("SEPARATE", 1), ("FUSED_KEY_VALUE", 2)),
        )

    def test_exact_state_topology_and_strict_loading_are_preserved(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(1501)
                model = _fixture_model(fixture)
                actual = tuple(
                    (name, tuple(value.shape), value.dtype)
                    for name, value in model.state_dict().items()
                )
                self.assertEqual(actual, EXPECTED_STATE_TOPOLOGY[fixture.name])

                torch.manual_seed(2501)
                restored = _fixture_model(fixture)
                self.assertTrue(
                    any(
                        not torch.equal(
                            restored.state_dict()[name],
                            value,
                        )
                        for name, value in model.state_dict().items()
                    )
                )
                load_result = restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(load_result.missing_keys, [])
                self.assertEqual(load_result.unexpected_keys, [])
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value)

                fixture_inputs = _fixture_inputs(fixture)
                model_inputs = _clone_inputs_preserving_aliases(fixture_inputs)
                restored_inputs = _clone_inputs_preserving_aliases(fixture_inputs)
                torch.manual_seed(2601)
                model_output, _model_weights, model_auxiliary_loss = model(
                    *model_inputs
                )
                torch.manual_seed(2601)
                restored_output, _restored_weights, restored_auxiliary_loss = restored(
                    *restored_inputs
                )
                torch.testing.assert_close(restored_output, model_output)
                if model_auxiliary_loss is None:
                    self.assertIsNone(restored_auxiliary_loss)
                else:
                    torch.testing.assert_close(
                        restored_auxiliary_loss,
                        model_auxiliary_loss,
                    )

                model_loss = model_output.square().sum()
                restored_loss = restored_output.square().sum()
                if model_auxiliary_loss is not None:
                    model_loss = model_loss + model_auxiliary_loss
                    restored_loss = restored_loss + restored_auxiliary_loss
                model_loss.backward()
                restored_loss.backward()
                for model_input, restored_input in zip(
                    model_inputs,
                    restored_inputs,
                    strict=True,
                ):
                    self.assertEqual(
                        restored_input.grad is None,
                        model_input.grad is None,
                    )
                    if model_input.grad is not None:
                        torch.testing.assert_close(
                            restored_input.grad,
                            model_input.grad,
                        )
                for (
                    model_parameter,
                    restored_parameter,
                ) in zip(
                    model.parameters(),
                    restored.parameters(),
                    strict=True,
                ):
                    self.assertEqual(
                        restored_parameter.grad is None,
                        model_parameter.grad is None,
                    )
                    if model_parameter.grad is not None:
                        torch.testing.assert_close(
                            restored_parameter.grad,
                            model_parameter.grad,
                        )

    def test_seeded_construction_rng_and_forward_payloads_are_preserved(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(1501)
                model = _fixture_model(fixture)
                digest = hashlib.sha256(
                    torch.random.get_rng_state().numpy().tobytes()
                ).hexdigest()
                self.assertEqual(digest, RNG_DIGESTS[fixture.name])

                query, key, value = _fixture_inputs(fixture)
                output, _weights, auxiliary_loss = model(query, key, value)
                expected = torch.tensor(
                    fixture.expected_output,
                    dtype=torch.float64,
                ).view(2, 1, 2)
                torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
                if auxiliary_loss is not None:
                    self.assertEqual(auxiliary_loss.shape, torch.Size([]))
                    self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_invalid_config_is_rejected_before_rng_consumption(self):
        for fixture in FIXTURES:
            config = _fixture_config(fixture)
            config.embedding_dim = 0
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(17)
                expected_next_values = torch.randn(8)

                torch.manual_seed(17)
                with self.assertRaises(ValueError):
                    config.build()
                actual_next_values = torch.randn(8)

                torch.testing.assert_close(actual_next_values, expected_next_values)

    def test_seeded_payloads_preserve_gradient_flow(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture.name), torch.random.fork_rng():
                torch.manual_seed(1501)
                model = _fixture_model(fixture)
                query, key, value = _fixture_inputs(fixture)
                output, _weights, auxiliary_loss = model(query, key, value)
                objective = output.square().mean()
                if auxiliary_loss is not None:
                    objective = objective + auxiliary_loss
                objective.backward()

                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(
                    any(
                        gradient is not None
                        and torch.isfinite(gradient).all()
                        and torch.count_nonzero(gradient)
                        for gradient in gradients
                    )
                )
                self.assertIsNotNone(query.grad)
                self.assertTrue(torch.isfinite(query.grad).all())


if __name__ == "__main__":
    unittest.main()
