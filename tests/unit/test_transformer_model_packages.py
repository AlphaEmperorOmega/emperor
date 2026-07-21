import contextlib
import io
import unittest
from dataclasses import fields, replace
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from emperor.attention._variants.independent.layer import IndependentAttention
from emperor.attention._variants.mixture.layer import MixtureOfAttentionHeads
from emperor.attention._variants.self_attention.layer import SelfAttention
from emperor.augmentations.adaptive_parameters import (
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters._diagonals.variants.standard import (
    StandardDynamicDiagonal,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.halting import StickBreaking
from emperor.layers import ActivationOptions, LastLayerBiasOptions, RecurrentLayer
from emperor.layers._composition.gate import LayerGate
from emperor.linears import LinearLayer
from emperor.memory._variants.gated_residual import GatedResidualDynamicMemory
from emperor.transformer import (
    TransformerAttentionOptions,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    TransformerFeedForwardOptions,
)
from models.config_overrides import (
    iter_supported_config_keys,
    print_config_options,
)
from models.transformer.expert_linear.config_builder import (
    TransformerExpertLinearConfigBuilder,
)
from models.transformer.expert_linear.model import Model as ExpertLinearModel
from models.transformer.expert_linear_adaptive.config_builder import (
    TransformerExpertLinearAdaptiveConfigBuilder,
)
from models.transformer.expert_linear_adaptive.model import (
    Model as ExpertLinearAdaptiveModel,
)
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear.model import Model as LinearModel
from models.transformer.linear_adaptive.config_builder import (
    TransformerLinearAdaptiveConfigBuilder,
)
from models.transformer.linear_adaptive.model import Model as LinearAdaptiveModel


class TestTransformerModelPackages(unittest.TestCase):
    def package_cases(self):
        return (
            (
                TransformerLinearConfigBuilder,
                LinearModel,
                SelfAttention,
                IndependentAttention,
                LinearLayer,
                False,
                False,
            ),
            (
                TransformerLinearAdaptiveConfigBuilder,
                LinearAdaptiveModel,
                SelfAttention,
                IndependentAttention,
                AdaptiveLinearLayer,
                True,
                False,
            ),
            (
                TransformerExpertLinearConfigBuilder,
                ExpertLinearModel,
                MixtureOfAttentionHeads,
                MixtureOfAttentionHeads,
                LinearLayer,
                False,
                True,
            ),
            (
                TransformerExpertLinearAdaptiveConfigBuilder,
                ExpertLinearAdaptiveModel,
                MixtureOfAttentionHeads,
                MixtureOfAttentionHeads,
                AdaptiveLinearLayer,
                True,
                True,
            ),
        )

    def preset(self, builder_type, *, adaptive=False, **options):
        builder_options = dict(
            batch_size=4,
            model_dim=16,
            source_sequence_length=7,
            target_sequence_length=6,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=32,
            dropout_probability=0.0,
        )
        if "Expert" in builder_type.__name__:
            builder_options.update(expert_num_experts=4, expert_top_k=2)
        if adaptive:
            builder_options["diagonal_option"] = StandardDynamicDiagonalConfig
        builder_options.update(options)
        return builder_type(**builder_options).build()

    def controller_options(self):
        options = dict(
            attn_num_layers=2,
            attn_gate_flag=True,
            attn_halting_flag=True,
            attn_memory_flag=True,
            attn_recurrent_flag=True,
            attn_recurrent_max_steps=2,
            attn_recurrent_gate_flag=True,
            attn_recurrent_halting_flag=True,
            ff_stack_hidden_dim=16,
            ff_gate_flag=True,
            ff_halting_flag=True,
            ff_memory_flag=True,
            ff_recurrent_flag=True,
            ff_recurrent_max_steps=2,
            ff_recurrent_gate_flag=True,
            ff_recurrent_halting_flag=True,
        )
        controller_stacks = (
            "attn_gate_stack",
            "attn_halting_stack",
            "attn_memory_stack",
            "attn_recurrent_gate_stack",
            "attn_recurrent_halting_stack",
            "ff_gate_stack",
            "ff_halting_stack",
            "ff_memory_stack",
            "ff_recurrent_gate_stack",
            "ff_recurrent_halting_stack",
        )
        for stack in controller_stacks:
            options[f"{stack}_independent_flag"] = True
            options[f"{stack}_hidden_dim"] = 12
            options[f"{stack}_num_layers"] = 2
            options[f"{stack}_activation"] = ActivationOptions.SIGMOID
            if "halting" in stack:
                options[f"{stack}_last_layer_bias_option"] = (
                    LastLayerBiasOptions.DISABLED
                )
        return options

    def test_configs_expose_complete_gpt_attention_and_feed_forward_groups(self):
        gpt_config = import_module("models.gpt.linear.config")
        expected = {
            key
            for key in iter_supported_config_keys(gpt_config)
            if key.startswith(("ATTN_", "FF_"))
        }
        expected.add("ATTN_ZERO_ATTENTION_FLAG")

        for package in (
            "linear",
            "linear_adaptive",
            "expert_linear",
            "expert_linear_adaptive",
        ):
            with self.subTest(package=package):
                config = import_module(f"models.transformer.{package}.config")
                actual = {
                    key
                    for key in iter_supported_config_keys(config)
                    if key.startswith(("ATTN_", "FF_"))
                }
                self.assertEqual(actual, expected)
                config_path = Path(config.__file__)
                config_source = config_path.read_text()
                self.assertIn("ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG", config_source)
                self.assertIn("FF_RECURRENT_HALTING_STACK_BIAS_FLAG", config_source)
                self.assertFalse((config_path.parent / "path").exists())

    def test_shared_path_types_preserve_constructor_fields_and_are_reexported(self):
        attention = replace(
            TransformerAttentionOptions(),
            projection_bias_flag=False,
            num_layers=3,
        )
        feed_forward = replace(
            TransformerFeedForwardOptions(),
            hidden_dim=96,
            num_layers=4,
            bias_flag=False,
        )
        self.assertTrue(
            {
                "num_heads",
                "projection_bias_flag",
                "add_key_value_bias_flag",
                "zero_attention_flag",
            }.issubset({field.name for field in fields(attention)})
        )
        self.assertTrue(
            {"hidden_dim", "num_layers"}.issubset(
                {field.name for field in fields(feed_forward)}
            )
        )
        self.assertFalse(attention.stack_options.bias_flag)
        self.assertEqual(attention.stack_options.num_layers, 3)
        self.assertEqual(feed_forward.stack_options.hidden_dim, 96)
        self.assertEqual(feed_forward.stack_options.num_layers, 4)
        self.assertFalse(feed_forward.stack_options.bias_flag)

        for package in (
            "linear",
            "linear_adaptive",
            "expert_linear",
            "expert_linear_adaptive",
        ):
            with self.subTest(package=package):
                runtime_options = import_module(
                    f"models.transformer.{package}.runtime_options"
                )
                self.assertIs(
                    runtime_options.TransformerAttentionOptions,
                    TransformerAttentionOptions,
                )
                self.assertIs(
                    runtime_options.TransformerFeedForwardOptions,
                    TransformerFeedForwardOptions,
                )

    def test_all_config_path_keys_map_in_unscoped_and_scoped_forms(self):
        for package in (
            "linear",
            "linear_adaptive",
            "expert_linear",
            "expert_linear_adaptive",
        ):
            config = import_module(f"models.transformer.{package}.config")
            runtime_defaults = import_module(
                f"models.transformer.{package}.runtime_defaults"
            )
            path_keys = [
                key
                for key in iter_supported_config_keys(config)
                if key.startswith(("ATTN_", "FF_"))
            ]
            for key in path_keys:
                value = getattr(config, key)
                with self.subTest(package=package, key=key, scope="broadcast"):
                    runtime_defaults.runtime_from_flat({key.lower(): value})
                if key.startswith("ATTN_"):
                    suffix = key.removeprefix("ATTN_").lower()
                    prefixes = (
                        "encoder_attn_",
                        "decoder_self_attn_",
                        "decoder_cross_attn_",
                    )
                else:
                    suffix = key.removeprefix("FF_").lower()
                    prefixes = ("encoder_ff_", "decoder_ff_")
                for prefix in prefixes:
                    with self.subTest(
                        package=package,
                        key=key,
                        scope=prefix,
                    ):
                        runtime_defaults.runtime_from_flat({f"{prefix}{suffix}": value})

    def test_presets_search_metadata_and_cli_use_shared_path_options(self):
        for package in (
            "linear",
            "linear_adaptive",
            "expert_linear",
            "expert_linear_adaptive",
        ):
            presets_module = import_module(f"models.transformer.{package}.presets")
            search_module = import_module(f"models.transformer.{package}.search_space")
            presets = presets_module.ExperimentPresets()
            for preset, definition in presets_module._PRESET_DEFINITIONS.items():
                with self.subTest(package=package, preset=preset.name):
                    self.assertIsNotNone(presets._preset(**definition.preset_values))

            attention_bias_locks = presets.locks_for_preset(
                presets_module.ExperimentPreset.ATTENTION_BIAS
            )
            for prefix in (
                "encoder_attn_",
                "decoder_self_attn_",
                "decoder_cross_attn_",
            ):
                self.assertIn(f"{prefix}bias_flag", attention_bias_locks)
                self.assertIn(
                    f"{prefix}add_key_value_bias_flag",
                    attention_bias_locks,
                )

            search_keys = {
                key
                for key, value in vars(search_module).items()
                if key.startswith("SEARCH_SPACE_") and isinstance(value, list)
            }
            self.assertIn("SEARCH_SPACE_ATTN_NUM_HEADS", search_keys)
            self.assertIn("SEARCH_SPACE_FF_STACK_HIDDEN_DIM", search_keys)
            self.assertFalse(
                any(
                    key.startswith(
                        (
                            "SEARCH_SPACE_ENCODER_ATTN_",
                            "SEARCH_SPACE_DECODER_SELF_ATTN_",
                            "SEARCH_SPACE_DECODER_CROSS_ATTN_",
                            "SEARCH_SPACE_ENCODER_FEED_FORWARD_",
                            "SEARCH_SPACE_DECODER_FEED_FORWARD_",
                        )
                    )
                    for key in search_keys
                )
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_config_options(f"transformer/{package}")
            listing = output.getvalue()
            self.assertIn("--attn-gate-stack-independent-flag", listing)
            self.assertIn(
                "--attn-recurrent-halting-stack-bias-flag",
                listing,
            )
            self.assertIn("--ff-memory-stack-hidden-dim", listing)
            self.assertIn("--ff-recurrent-gate-stack-num-layers", listing)

    def test_path_broadcast_scoped_precedence_legacy_aliases_and_errors(self):
        for builder_type, *_ in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                runtime = builder_type(
                    attn_num_heads=8,
                    attn_stack_hidden_dim=24,
                    attn_gate_flag=True,
                    attn_gate_stack_independent_flag=True,
                    attn_gate_stack_hidden_dim=20,
                    ff_stack_hidden_dim=48,
                    ff_recurrent_flag=True,
                    encoder_attn_num_heads=2,
                    decoder_ff_stack_hidden_dim=40,
                ).runtime
                self.assertEqual(runtime.encoder_attention_options.num_heads, 2)
                self.assertEqual(
                    runtime.decoder_self_attention_options.num_heads,
                    8,
                )
                self.assertEqual(
                    runtime.decoder_cross_attention_options.num_heads,
                    8,
                )
                for attention in (
                    runtime.encoder_attention_options,
                    runtime.decoder_self_attention_options,
                    runtime.decoder_cross_attention_options,
                ):
                    self.assertEqual(attention.stack_options.hidden_dim, 24)
                    self.assertTrue(attention.layer_controller_options.stack_gate_flag)
                    self.assertEqual(
                        attention.layer_controller_options.gate_stack_options.hidden_dim,
                        20,
                    )
                self.assertEqual(
                    runtime.encoder_feed_forward_options.hidden_dim,
                    48,
                )
                self.assertEqual(
                    runtime.decoder_feed_forward_options.hidden_dim,
                    40,
                )
                self.assertTrue(
                    runtime.encoder_feed_forward_options.recurrent_controller_options.recurrent_flag
                )

                inherited_globals = builder_type(
                    model_dim=20,
                    dropout_probability=0.25,
                ).runtime
                for attention in (
                    inherited_globals.encoder_attention_options,
                    inherited_globals.decoder_self_attention_options,
                    inherited_globals.decoder_cross_attention_options,
                ):
                    self.assertEqual(attention.stack_options.hidden_dim, 20)
                for feed_forward in (
                    inherited_globals.encoder_feed_forward_options,
                    inherited_globals.decoder_feed_forward_options,
                ):
                    self.assertEqual(
                        feed_forward.stack_options.dropout_probability,
                        0.25,
                    )

                legacy = builder_type(
                    feed_forward_hidden_dim=36,
                    encoder_feed_forward_hidden_dim=28,
                    attn_projection_bias_flag=False,
                ).runtime
                self.assertEqual(legacy.encoder_feed_forward_options.hidden_dim, 28)
                self.assertEqual(legacy.decoder_feed_forward_options.hidden_dim, 36)
                self.assertFalse(legacy.encoder_attention_options.projection_bias_flag)

                with self.assertRaisesRegex(ValueError, "Conflicting values"):
                    builder_type(
                        ff_stack_hidden_dim=48,
                        feed_forward_hidden_dim=32,
                    )
                with self.assertRaisesRegex(ValueError, "Conflicting values"):
                    builder_type(
                        attn_bias_flag=True,
                        attn_projection_bias_flag=False,
                    )
                with self.assertRaises(TypeError):
                    builder_type(not_a_transformer_option=True)

    def test_all_path_controllers_build_forward_and_receive_gradients(self):
        for (
            builder_type,
            model_type,
            _self_attention_type,
            _cross_attention_type,
            _leaf_type,
            adaptive,
            _expert,
        ) in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                model = model_type(
                    self.preset(
                        builder_type,
                        **self.controller_options(),
                    )
                )
                source = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
                target = torch.tensor([[2, 11, 3], [2, 12, 3]])

                logits, auxiliary_loss = model(source, target)
                (logits[..., :16].square().mean() + auxiliary_loss).backward()

                self.assertEqual(logits.shape, (2, 3, 8192))
                for controller_type in (
                    LayerGate,
                    StickBreaking,
                    GatedResidualDynamicMemory,
                    RecurrentLayer,
                ):
                    controllers = [
                        module
                        for module in model.modules()
                        if isinstance(module, controller_type)
                    ]
                    with self.subTest(
                        package=builder_type.__name__,
                        controller=controller_type.__name__,
                    ):
                        self.assertTrue(controllers)
                        self.assertTrue(
                            any(
                                parameter.grad is not None
                                for controller in controllers
                                for parameter in controller.parameters()
                                if parameter.requires_grad
                            )
                        )
                if adaptive:
                    ordinary_controller_modules = [
                        module
                        for module in model.modules()
                        if isinstance(
                            module,
                            (LayerGate, StickBreaking, GatedResidualDynamicMemory),
                        )
                    ]
                    for controller in ordinary_controller_modules:
                        with self.subTest(
                            package=builder_type.__name__,
                            ordinary_controller=type(controller).__name__,
                        ):
                            self.assertTrue(
                                any(
                                    isinstance(module, LinearLayer)
                                    for module in controller.modules()
                                )
                            )
                            self.assertFalse(
                                any(
                                    isinstance(module, AdaptiveLinearLayer)
                                    for module in controller.modules()
                                )
                            )

    def test_scoped_encoder_and_decoder_overrides_are_independent(self):
        for builder_type, *_ in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                options = dict(
                    encoder_num_layers=1,
                    decoder_num_layers=2,
                    encoder_attn_num_heads=2,
                    decoder_self_attn_num_heads=4,
                    decoder_cross_attn_num_heads=1,
                    encoder_feed_forward_hidden_dim=24,
                    decoder_feed_forward_hidden_dim=40,
                )
                config = self.preset(builder_type, **options)
                experiment_config = config.experiment_config
                encoder_layer = (
                    experiment_config.encoder_config.layer_config.layer_model_config
                )
                decoder_layer = (
                    experiment_config.decoder_config.layer_config.layer_model_config
                )

                self.assertEqual(experiment_config.encoder_config.num_layers, 1)
                self.assertEqual(experiment_config.decoder_config.num_layers, 2)
                self.assertEqual(encoder_layer.attention_config.num_heads, 2)
                self.assertEqual(decoder_layer.self_attention_config.num_heads, 4)
                self.assertEqual(decoder_layer.cross_attention_config.num_heads, 1)
                encoder_feed_forward_stack = (
                    encoder_layer.feed_forward_config.stack_config
                )
                decoder_feed_forward_stack = (
                    decoder_layer.feed_forward_config.stack_config
                )
                if hasattr(encoder_feed_forward_stack, "stack_config"):
                    encoder_feed_forward_stack = encoder_feed_forward_stack.stack_config
                    decoder_feed_forward_stack = decoder_feed_forward_stack.stack_config
                self.assertEqual(
                    encoder_feed_forward_stack.hidden_dim,
                    24,
                )
                self.assertEqual(
                    decoder_feed_forward_stack.hidden_dim,
                    40,
                )

    def test_backend_types_forward_contract_weight_tying_and_gradients(self):
        for (
            builder_type,
            model_type,
            self_attention_type,
            cross_attention_type,
            leaf_type,
            adaptive,
            expert,
        ) in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                model = model_type(self.preset(builder_type, adaptive=adaptive))
                encoder_layer = next(
                    module
                    for module in model.encoder.modules()
                    if isinstance(module, TransformerEncoderLayer)
                )
                decoder_layer = next(
                    module
                    for module in model.decoder.modules()
                    if isinstance(module, TransformerDecoderLayer)
                )
                self.assertIsInstance(
                    encoder_layer.self_attention_model,
                    self_attention_type,
                )
                self.assertIsInstance(
                    decoder_layer.self_attention_model,
                    self_attention_type,
                )
                self.assertIsInstance(
                    decoder_layer.cross_attention_model,
                    cross_attention_type,
                )
                encoder_attention_parameter = next(
                    encoder_layer.self_attention_model.parameters()
                )
                decoder_self_attention_parameter = next(
                    decoder_layer.self_attention_model.parameters()
                )
                decoder_cross_attention_parameter = next(
                    decoder_layer.cross_attention_model.parameters()
                )
                self.assertIsNot(
                    encoder_attention_parameter,
                    decoder_self_attention_parameter,
                )
                self.assertIsNot(
                    decoder_self_attention_parameter,
                    decoder_cross_attention_parameter,
                )
                self.assertIsNot(
                    next(encoder_layer.feed_forward_model.parameters()),
                    next(decoder_layer.feed_forward_model.parameters()),
                )
                self.assertTrue(
                    any(
                        isinstance(module, leaf_type)
                        for module in encoder_layer.self_attention_model.modules()
                    )
                )
                self.assertTrue(
                    any(
                        isinstance(module, leaf_type)
                        for module in decoder_layer.cross_attention_model.modules()
                    )
                )
                if expert:
                    self.assertTrue(
                        any(
                            isinstance(module, MixtureOfExperts)
                            for module in encoder_layer.feed_forward_model.modules()
                        )
                    )
                else:
                    self.assertTrue(
                        any(
                            isinstance(module, leaf_type)
                            for module in encoder_layer.feed_forward_model.modules()
                        )
                    )

                source = torch.tensor(
                    [
                        [2, 8, 9, 3, 0],
                        [2, 10, 3, 0, 0],
                        [2, 11, 12, 13, 3],
                    ]
                )
                target = torch.tensor([[2, 14, 3, 0], [2, 15, 3, 0], [2, 16, 17, 3]])
                logits, auxiliary_loss = model(source, target)
                self.assertEqual(logits.shape, (3, 4, 8192))
                self.assertEqual(auxiliary_loss.ndim, 0)
                self.assertIs(model.source_embedding, model.target_embedding)
                self.assertIs(
                    model.output_projection.weight,
                    model.shared_embedding.weight,
                )
                (logits[..., :32].square().mean() + auxiliary_loss).backward()
                for name, module in (
                    ("embedding", model.shared_embedding),
                    ("encoder", encoder_layer),
                    ("decoder-self", decoder_layer.self_attention_model),
                    ("decoder-cross", decoder_layer.cross_attention_model),
                    ("decoder-feed-forward", decoder_layer.feed_forward_model),
                ):
                    with self.subTest(package=builder_type.__name__, path=name):
                        self.assertTrue(
                            any(
                                parameter.grad is not None
                                for parameter in module.parameters()
                                if parameter.requires_grad
                            )
                        )
                if adaptive:
                    adaptive_modules = [
                        module
                        for module in model.modules()
                        if isinstance(module, StandardDynamicDiagonal)
                    ]
                    self.assertTrue(adaptive_modules)
                    self.assertTrue(
                        any(
                            parameter.grad is not None
                            for module in adaptive_modules
                            for parameter in module.parameters()
                        )
                    )

    def test_forward_validation_rejects_invalid_translation_ids(self):
        for builder_type, model_type, *_rest in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                model = model_type(
                    self.preset(
                        builder_type,
                        vocab_size=32,
                    )
                )
                source = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
                target = torch.tensor([[2, 11, 3], [2, 12, 3]])
                cases = (
                    (([[2, 8, 3]], target), TypeError),
                    ((torch.tensor([2, 8, 3]), target), ValueError),
                    ((torch.empty((0, 3), dtype=torch.long), target), ValueError),
                    ((torch.empty((2, 0), dtype=torch.long), target), ValueError),
                    ((source.float(), target), TypeError),
                    ((torch.tensor([[2, -1, 3], [2, 4, 3]]), target), ValueError),
                    ((torch.tensor([[2, 32, 3], [2, 4, 3]]), target), ValueError),
                    ((torch.ones((2, 8), dtype=torch.long), target), ValueError),
                    ((source, torch.ones((1, 3), dtype=torch.long)), ValueError),
                    ((source, target.float()), TypeError),
                )
                for arguments, error in cases:
                    with self.subTest(
                        package=builder_type.__name__,
                        error=error.__name__,
                    ):
                        with self.assertRaises(error):
                            model(*arguments)

                logits, auxiliary_loss = model(source.int(), target.short())
                self.assertEqual(logits.shape, (2, 3, 32))
                self.assertEqual(auxiliary_loss.shape, ())

    def test_nested_path_construction_validation_has_stable_errors(self):
        common_cases = (
            {"attn_num_layers": 0},
            {"attn_stack_hidden_dim": 0},
            {"attn_stack_dropout_probability": 1.1},
            {"attn_recurrent_max_steps": 0},
            {"attn_gate_stack_hidden_dim": 0},
            {"attn_halting_stack_num_layers": 0},
            {"ff_num_layers": 0},
            {"ff_stack_hidden_dim": 0},
            {"ff_stack_dropout_probability": -0.1},
            {"ff_recurrent_max_steps": 0},
            {"ff_memory_stack_hidden_dim": 0},
            {"ff_recurrent_halting_stack_num_layers": 0},
            {"encoder_recurrent_max_steps": 0},
            {"decoder_recurrent_max_steps": 0},
            {"model_dim": 16, "decoder_cross_attn_num_heads": 3},
        )
        for builder_type, *_rest in self.package_cases():
            for options in common_cases:
                with self.subTest(
                    package=builder_type.__name__,
                    options=options,
                ):
                    with self.assertRaises((TypeError, ValueError)):
                        builder_type(**options)

            valid_runtime = builder_type().runtime
            with self.assertRaisesRegex(ValueError, "batch_size"):
                builder_type(runtime=replace(valid_runtime, batch_size=0))

            if "Expert" in builder_type.__name__:
                expert_cases = (
                    {"attention_expert_num_experts": 0},
                    {"feed_forward_expert_top_k": 0},
                    {
                        "attention_expert_num_experts": 2,
                        "attention_expert_top_k": 3,
                    },
                    {"attention_expert_switch_loss_weight": -0.1},
                    {"feed_forward_expert_capacity_factor": -0.1},
                    {
                        "attention_expert_num_experts": 2,
                        "attention_expert_top_k": 2,
                        "attention_expert_capacity_factor": 1.0,
                    },
                )
                for options in expert_cases:
                    with self.subTest(
                        package=builder_type.__name__,
                        options=options,
                    ):
                        with self.assertRaises((TypeError, ValueError)):
                            builder_type(**options)

    def test_decoder_causality_blocks_future_target_changes(self):
        for builder_type, model_type, *_prefix, expert in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                torch.manual_seed(0)
                model = model_type(
                    self.preset(
                        builder_type,
                        vocab_size=32,
                        dropout_probability=0.0,
                    )
                ).eval()
                source = torch.tensor([[2, 8, 9, 3], [2, 10, 11, 3]])
                target = torch.tensor([[2, 12, 13, 3], [2, 14, 15, 3]])
                changed = target.clone()
                changed[:, -1] = torch.tensor([20, 21])

                with torch.no_grad():
                    original, _ = model(source, target)
                    modified, _ = model(source, changed)

                torch.testing.assert_close(
                    original[:, :-1],
                    modified[:, :-1],
                    atol=1e-5 if expert else 1e-6,
                    rtol=1e-5 if expert else 1e-6,
                )

    def test_source_target_and_encoder_padding_masks_propagate(self):
        for builder_type, model_type, *_rest in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                model = model_type(
                    self.preset(
                        builder_type,
                        vocab_size=32,
                    )
                )
                encoder = _StateSpy()
                decoder = _StateSpy()
                model.encoder = encoder
                model.decoder = decoder
                source = torch.tensor([[2, 8, 3, 0], [2, 9, 0, 0]])
                target = torch.tensor([[2, 11, 3, 0], [2, 12, 0, 0]])

                model(source, target)

                torch.testing.assert_close(
                    encoder.state.key_padding_mask,
                    source.eq(model.pad_token_id),
                )
                torch.testing.assert_close(
                    decoder.state.target_key_padding_mask,
                    target.eq(model.pad_token_id),
                )
                torch.testing.assert_close(
                    decoder.state.encoder_padding_mask,
                    source.eq(model.pad_token_id),
                )

    def test_auxiliary_loss_is_scalar_added_and_backpropagated(self):
        for builder_type, model_type, *_rest in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                model = model_type(
                    self.preset(
                        builder_type,
                        vocab_size=32,
                    )
                )
                encoder = _AuxiliaryState(0.1)
                decoder = _AuxiliaryState(0.2)
                model.encoder = encoder
                model.decoder = decoder
                source = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
                target = torch.tensor([[2, 11, 3, 0], [2, 12, 13, 3]])

                output = model._model_step_outputs((source, target))
                expected_task_loss = model.loss_fn(
                    output.logits.reshape(-1, output.logits.size(-1)),
                    output.labels.reshape(-1),
                )
                torch.testing.assert_close(
                    output.auxiliary_loss,
                    output.total_loss.new_tensor(0.3),
                )
                torch.testing.assert_close(
                    output.total_loss,
                    expected_task_loss + output.auxiliary_loss,
                )
                output.total_loss.backward()
                self.assertIsNotNone(encoder.value.grad)
                self.assertIsNotNone(decoder.value.grad)

    def test_outer_halting_controllers_disable_binary_head_bias(self):
        for builder_type, *_rest in self.package_cases():
            for recurrent in (False, True):
                with self.subTest(
                    package=builder_type.__name__,
                    recurrent=recurrent,
                ):
                    cfg = self.preset(
                        builder_type,
                        encoder_num_layers=2,
                        decoder_num_layers=2,
                        stack_halting_flag=not recurrent,
                        recurrent_flag=recurrent,
                        recurrent_halting_flag=recurrent,
                    )
                    encoder = cfg.experiment_config.encoder_config
                    halting = (
                        encoder.halting_config
                        if recurrent
                        else encoder.shared_halting_config
                    )
                    controller = halting.halting_gate_config
                    self.assertEqual(controller.output_dim, 2)
                    self.assertEqual(
                        controller.last_layer_bias_option,
                        LastLayerBiasOptions.DISABLED,
                    )

    def test_path_halting_controllers_disable_binary_head_bias(self):
        for builder_type, *_rest in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                cfg = self.preset(
                    builder_type,
                    attn_halting_flag=True,
                    ff_halting_flag=True,
                )
                encoder = cfg.experiment_config.encoder_config
                encoder = getattr(encoder, "block_config", encoder)
                layer = encoder.layer_config.layer_model_config
                attention_stack = layer.attention_config.projection_model_config
                attention_stack = getattr(
                    attention_stack,
                    "block_config",
                    attention_stack,
                )
                feed_forward_stack = layer.feed_forward_config.stack_config
                feed_forward_stack = getattr(
                    feed_forward_stack,
                    "stack_config",
                    feed_forward_stack,
                )
                for stack in (attention_stack, feed_forward_stack):
                    controller = stack.shared_halting_config.halting_gate_config
                    self.assertEqual(controller.output_dim, 2)
                    self.assertEqual(
                        controller.last_layer_bias_option,
                        LastLayerBiasOptions.DISABLED,
                    )

    def test_generation_is_deterministic_and_restores_mode_on_failure(self):
        for builder_type, model_type, *_rest in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                torch.manual_seed(0)
                model = model_type(
                    self.preset(
                        builder_type,
                        vocab_size=32,
                        dropout_probability=0.5,
                    )
                )
                source = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
                model.train()
                first = model.generate(source, max_length=5)
                second = model.generate(source, max_length=5)
                torch.testing.assert_close(first, second)
                self.assertTrue(model.training)

                model.eval()
                model.generate(source, max_length=1)
                self.assertFalse(model.training)

                model.train()
                with (
                    patch.object(
                        model,
                        "_decode",
                        side_effect=RuntimeError("generation failed"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "generation failed"),
                ):
                    model.generate(source, max_length=5)
                self.assertTrue(model.training)

    def test_generation_validates_requests_and_stops_each_sample(self):
        for builder_type, model_type, *_rest in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                model = model_type(
                    self.preset(
                        builder_type,
                        vocab_size=32,
                    )
                )
                source = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3], [2, 11, 3, 0]])
                model.decoder = _StateSpy()
                model.output_projection = _ScheduledProjection(
                    32,
                    model.eos_token_id,
                )
                generated = model.generate(source, max_length=5)
                expected = torch.tensor(
                    [
                        [model.bos_token_id, model.eos_token_id, 0, 0, 0],
                        [model.bos_token_id, 4, model.eos_token_id, 0, 0],
                        [model.bos_token_id, 5, 6, model.eos_token_id, 0],
                    ]
                )
                torch.testing.assert_close(generated.cpu(), expected)

                for invalid in (True, 1.5, "4"):
                    with self.subTest(package=builder_type.__name__, invalid=invalid):
                        with self.assertRaises(TypeError):
                            model.generate(source, max_length=invalid)
                for invalid in (0, model.target_sequence_length + 1):
                    with self.assertRaises(ValueError):
                        model.generate(source, max_length=invalid)
                with self.assertRaises(TypeError):
                    model.generate([[2, 8, 3]])
                with self.assertRaises(ValueError):
                    model.generate(torch.empty((1, 0), dtype=torch.long))

    def test_generation_caches_encoder_stops_at_eos_and_pads_remainder(self):
        for builder_type, model_type, *_rest in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                model = model_type(self.preset(builder_type))
                source = torch.tensor(
                    [
                        [2, 8, 3, 0],
                        [2, 9, 10, 3],
                        [2, 11, 3, 0],
                    ]
                )
                with torch.no_grad():
                    model.shared_embedding.weight.zero_()
                    model.shared_embedding.weight[model.eos_token_id].fill_(1.0)

                model_dim = model.experiment_config.model_dim

                def decoder_output(
                    target_ids,
                    encoder_output,
                    source_mask,
                    model_dim=model_dim,
                ):
                    hidden = torch.ones(
                        target_ids.size(0),
                        target_ids.size(1),
                        model_dim,
                        device=target_ids.device,
                    )
                    return hidden, hidden.new_zeros(())

                with (
                    patch.object(
                        model,
                        "_encode",
                        wraps=model._encode,
                    ) as encode,
                    patch.object(
                        model,
                        "_decode",
                        side_effect=decoder_output,
                    ),
                ):
                    generated = model.generate(source, max_length=6)

                self.assertEqual(encode.call_count, 1)
                self.assertEqual(generated.shape, (3, 6))
                self.assertTrue(torch.all(generated[:, 0] == model.bos_token_id))
                self.assertTrue(torch.all(generated[:, 1] == model.eos_token_id))
                self.assertTrue(torch.all(generated[:, 2:] == model.pad_token_id))


class _StateSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None

    def forward(self, state):
        self.state = state
        state.loss = state.hidden.new_zeros(())
        return state


class _AuxiliaryState(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, state):
        state.loss = self.value
        return state


class _ScheduledProjection(nn.Module):
    def __init__(self, vocab_size: int, eos_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.call_index = 0

    def forward(self, hidden):
        schedules = (
            (self.eos_token_id, 4, 5),
            (self.eos_token_id, self.eos_token_id, 6),
            (self.eos_token_id, self.eos_token_id, self.eos_token_id),
        )
        token_ids = schedules[min(self.call_index, len(schedules) - 1)]
        self.call_index += 1
        logits = hidden.new_zeros((hidden.size(0), self.vocab_size))
        logits[torch.arange(hidden.size(0)), torch.tensor(token_ids)] = 1.0
        return logits


if __name__ == "__main__":
    unittest.main()
