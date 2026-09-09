"""Package selection must reach real adaptive layers, including bank-only paths."""

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    DiagonallyModulatedLowRankDynamicWeightConfig,
    LowRankFactorSourceOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    WeightDecayScheduleOptions,
)
from models.catalog import model_package


@pytest.mark.parametrize(
    "package_name",
    ["transformer/linear_adaptive", "transformer/expert_linear_adaptive"],
)
def test_transformer_generation_options_preserve_path_precedence(package_name):
    package = model_package(package_name)
    overrides = dict(
        weight_option=DiagonallyModulatedLowRankDynamicWeightConfig,
        weight_input_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
        encoder_ff_adaptive_weight_output_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
        encoder_ff_adaptive_bias_option=MatrixBiasMixtureConfig,
        encoder_ff_adaptive_bias_mixture_num_experts=3,
        encoder_ff_adaptive_bias_mixture_top_k=2,
        encoder_ff_adaptive_weight_coefficient_generator_stack_independent_flag=True,
        encoder_ff_adaptive_weight_coefficient_generator_stack_hidden_dim=5,
    )
    runtime = package.bind_runtime_defaults(overrides)
    options = runtime.encoder_feed_forward_adaptive_options
    assert (
        options.weight_input_factor_source
        is LowRankFactorSourceOptions.SHARED_PARAMETER
    )
    assert (
        options.weight_output_factor_source
        is LowRankFactorSourceOptions.SHARED_PARAMETER
    )
    assert options.bias_mixture_num_experts == 3
    assert options.weight_coefficient_generator_stack_options.hidden_dim == 5


@pytest.mark.parametrize(
    "weight_option",
    [DiagonallyModulatedLowRankDynamicWeightConfig, MatrixWeightsMixtureConfig],
)
def test_linear_package_builds_bank_and_mixed_factor_sources_from_scalars(
    weight_option,
):
    from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig

    AdaptiveLinearLayer = AdaptiveLinearLayerConfig().registry_owner()
    from models.linears.linear_adaptive.config_builder import (
        LinearAdaptiveConfigBuilder,
    )

    package = model_package("linears/linear_adaptive")
    runtime = package.bind_runtime_defaults(
        dict(
            input_dim=2,
            hidden_dim=4,
            output_dim=3,
            weight_option_flag=True,
            weight_option=weight_option,
            weight_mixture_num_experts=3,
            weight_mixture_top_k=2,
            weight_decay_schedule=WeightDecayScheduleOptions.LINEAR,
            weight_decay_rate=0.25,
            weight_decay_warmup_batches=2,
            bias_decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
            bias_decay_rate=0.5,
            bias_decay_warmup_batches=1,
            weight_input_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
            bias_option=MatrixBiasMixtureConfig,
            bias_option_flag=True,
            bias_mixture_num_experts=3,
            bias_mixture_top_k=2,
            weight_coefficient_generator_stack_independent_flag=True,
            weight_coefficient_generator_stack_hidden_dim=5,
        )
    )
    configuration = LinearAdaptiveConfigBuilder(runtime=runtime).build()
    model = package.build_model(configuration)
    adaptive = [
        m
        for m in model.modules()
        if isinstance(m, AdaptiveLinearLayer) and m.adaptive_behaviour is not None
    ]
    assert adaptive
    assert any(
        isinstance(
            layer.adaptive_behaviour.bias_model,
            MatrixBiasMixtureConfig().registry_owner(),
        )
        for layer in adaptive
    )
    for layer in adaptive:
        if not isinstance(
            layer.adaptive_behaviour.bias_model,
            MatrixBiasMixtureConfig().registry_owner(),
        ):
            continue
        weights = layer.adaptive_behaviour.weight_model
        if weight_option is DiagonallyModulatedLowRankDynamicWeightConfig:
            assert weights.input_factor is not None and weights.output_factor is None
            assert weights.coefficient_model.hidden_dim == 5
            assert weights.coefficient_model.output_dim == weights.rank
        assert weights.cfg.decay_schedule is WeightDecayScheduleOptions.LINEAR
        assert weights.cfg.decay_rate == 0.25 and weights.cfg.decay_warmup_batches == 2
        biases = layer.adaptive_behaviour.bias_model
        assert biases.cfg.decay_schedule is WeightDecayScheduleOptions.MULTIPLICATIVE
        assert biases.cfg.decay_rate == 0.5 and biases.cfg.decay_warmup_batches == 1
        assert layer.weight_params is not None and layer.bias_params is not None
        output = layer(torch.randn(2, layer.input_dim))
        assert output.shape == (2, layer.output_dim)
        output.square().sum().backward()


@pytest.mark.parametrize(
    "key", ["transformer/linear_adaptive", "transformer/expert_linear_adaptive"]
)
@pytest.mark.parametrize("combined", [False, True])
def test_transformer_bank_only_and_combined_build_forward_backward_inspection(
    key, combined
):
    import importlib

    from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig

    AdaptiveLinearLayer = AdaptiveLinearLayerConfig().registry_owner()
    from model_runtime.inspection import configuration_schema
    from model_runtime.packages import parse_config_value, serialize_config_value

    package = model_package(key)
    overrides = dict(
        batch_size=2,
        vocab_size=16,
        model_dim=4,
        source_sequence_length=2,
        target_sequence_length=2,
        encoder_num_layers=1,
        decoder_num_layers=1,
        attn_num_heads=2,
        ff_stack_hidden_dim=4,
        dropout_probability=0.0,
        weight_decay_schedule=WeightDecayScheduleOptions.LINEAR,
        weight_decay_rate=0.25,
        weight_decay_warmup_batches=2,
        bias_decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
        bias_decay_rate=0.5,
        bias_decay_warmup_batches=1,
        weight_option_flag=True,
        diagonal_option_flag=False,
        mask_option_flag=False,
        weight_option=DiagonallyModulatedLowRankDynamicWeightConfig
        if combined
        else MatrixWeightsMixtureConfig,
        weight_mixture_num_experts=3,
        weight_mixture_top_k=2,
        weight_input_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
        bias_option=MatrixBiasMixtureConfig,
        bias_option_flag=True,
        bias_mixture_num_experts=2,
        bias_mixture_top_k=1,
        weight_coefficient_generator_stack_independent_flag=True,
        weight_coefficient_generator_stack_hidden_dim=5,
        bias_mixture_router_generator_stack_independent_flag=True,
        bias_mixture_router_generator_stack_hidden_dim=3,
    )
    configuration = package.build_configuration(config_overrides=overrides)
    model = package.build_model(configuration)
    layers = [
        m
        for m in model.modules()
        if isinstance(m, AdaptiveLinearLayer) and m.adaptive_behaviour is not None
    ]
    assert layers and all(
        m.weight_params is not None and m.bias_params is not None for m in layers
    )
    for layer in layers:
        weights = layer.adaptive_behaviour.weight_model
        biases = layer.adaptive_behaviour.bias_model
        assert weights.cfg.decay_schedule is WeightDecayScheduleOptions.LINEAR
        assert weights.cfg.decay_rate == 0.25 and weights.cfg.decay_warmup_batches == 2
        assert biases.cfg.decay_schedule is WeightDecayScheduleOptions.MULTIPLICATIVE
        assert biases.cfg.decay_rate == 0.5 and biases.cfg.decay_warmup_batches == 1
    assert all(
        isinstance(
            m.adaptive_behaviour.weight_model,
            DiagonallyModulatedLowRankDynamicWeightConfig().registry_owner(),
        )
        == combined
        for m in layers
    )
    executed = set()
    output_handles = []

    def observe_backward_use(module, _inputs, output):
        if output.requires_grad:
            output_handles.append(
                output.register_hook(lambda _gradient: executed.add(module))
            )

    handles = [layer.register_forward_hook(observe_backward_use) for layer in layers]
    ids = torch.tensor([[2, 3], [2, 3]])
    output, loss = model(ids, ids)
    assert output.shape == (2, 2, 16) and torch.isfinite(output).all()
    for handle in handles:
        handle.remove()
    (output.square().mean() + loss).backward()
    assert all(
        m.adaptive_behaviour.bias_model.parameter_bank.grad is not None
        for m in executed
    )
    assert executed
    for handle in output_handles:
        handle.remove()
    config_module = importlib.import_module(
        "models." + key.replace("/", ".") + ".config"
    )
    schema = {field.key: field for field in configuration_schema(package).fields}
    for field, value in [
        ("WEIGHT_INPUT_FACTOR_SOURCE", LowRankFactorSourceOptions.SHARED_PARAMETER),
        ("BIAS_OPTION", MatrixBiasMixtureConfig),
    ]:
        encoded = serialize_config_value(value)
        assert parse_config_value(config_module, field, encoded) is value
        assert encoded in schema[field].choices
