from __future__ import annotations

from typing import get_args

import pytest
import torch

from emperor.layers import (
    LayerNormPositionOptions,
    NormalizationOptions,
    WeightedResidualConfig,
)
from model_runtime.inspection import configuration_schema
from models.catalog import discover_model_packages, model_package

_PACKAGES = tuple(
    package.catalog_key for package in discover_model_packages()
    if package.identity.model_type in ('linears', 'transformer', 'bert', 'gpt', 'vit', 'experts')
)
_NORMALIZATION_TYPES = {
    NormalizationOptions.RMS_NORM: "RMSNorm",
    NormalizationOptions.LAYER_NORM: "LayerNorm",
    NormalizationOptions.DYNAMIC_TANH: "DynamicTanh",
    NormalizationOptions.DERF: "DynamicErf",
    NormalizationOptions.DYISRU: "DynamicISRU",
}


def _selectors(package):
    spec = package.runtime_defaults_spec
    return tuple(
        key
        for key in spec.supported_keys
        if spec.annotations.get(key) is NormalizationOptions
        or NormalizationOptions in get_args(spec.annotations.get(key))
    )


def _small_overrides(package):
    family = package.identity.model_type
    proposed = {"batch_size": 2, "hidden_dim": 8, "stack_num_layers": 2}
    if family in ("bert", "gpt"):
        proposed.update(
            input_dim=32,
            output_dim=32,
            sequence_length=4,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
        )
    elif family == "transformer":
        proposed.update(
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
        )
    elif family == "parametric":
        proposed.update(
            stack_residual_connection_option=WeightedResidualConfig,
            stack_residual_model_flag=True,
            residual_stack_independent_flag=True,
            residual_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            residual_stack_num_layers=2,
            residual_stack_hidden_dim=8,
        )
    supported = {key.lower() for key in package.runtime_defaults_spec.supported_keys}
    return {key: value for key, value in proposed.items() if key in supported}


def _forward(package, model):
    family = package.identity.model_type
    if family in ("bert", "gpt", "transformer"):
        tokens = torch.tensor([[2, 7, 8, 3], [2, 9, 10, 3]])
        return model(tokens, tokens) if family == "transformer" else model(tokens)
    dataset = package.resolve_dataset(None)
    return model(
        torch.randn(
            2, dataset.num_channels, dataset.default_height, dataset.default_width
        )
    )


@pytest.mark.parametrize("catalog_key", _PACKAGES)
def test_model_schema_exposes_normalization_next_to_every_position(catalog_key):
    package = model_package(catalog_key)
    spec = package.runtime_defaults_spec
    fields = {field.key: field for field in configuration_schema(package).fields}
    selectors = _selectors(package)
    assert selectors
    for key in spec.supported_keys:
        if "LAYER_NORM_POSITION" not in key:
            continue
        selector = key.replace("LAYER_NORM_POSITION", "NORMALIZATION")
        assert selector in selectors
        assert (selector in fields) == (key in fields)
        assert (
            spec.configuration_metadata[selector]["sectionPath"]
            == (spec.configuration_metadata[key]["sectionPath"])
        )


@pytest.mark.parametrize("catalog_key", _PACKAGES)
@pytest.mark.parametrize("normalization", tuple(NormalizationOptions))
def test_every_model_runs_and_backpropagates_with_each_normalization(
    catalog_key,
    normalization,
):
    torch.manual_seed(43)
    package = model_package(catalog_key)
    overrides = _small_overrides(package)
    overrides.update({key.lower(): normalization for key in _selectors(package)})
    configuration = package.build_configuration(config_overrides=overrides)
    model = package.build_model(configuration)
    norm_modules = {
        name: module
        for name, module in model.named_modules()
        if type(module).__name__ in _NORMALIZATION_TYPES.values()
    }
    assert norm_modules, catalog_key
    for name, module in norm_modules.items():
        assert type(module).__name__ == _NORMALIZATION_TYPES[normalization], name
    visited = set()
    handles = [
        module.register_forward_hook(lambda module, args, result: visited.add(module))
        for module in norm_modules.values()
    ]
    try:
        outputs = _forward(package, model)
    finally:
        for handle in handles:
            handle.remove()
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
    loss = sum(tensor.float().square().mean() for tensor in tensors)
    loss.backward()
    assert visited
    norm_gradients = [
        parameter.grad
        for module in visited
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert norm_gradients
    assert all(torch.isfinite(gradient).all() for gradient in norm_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in norm_gradients)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    # Same option must reconstruct the checkpoint's learned normalization parameters.
    restored = package.build_model(configuration)
    restored.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("catalog_key", _PACKAGES)
def test_model_runtime_rejects_invalid_normalization(catalog_key):
    package = model_package(catalog_key)
    selector = _selectors(package)[0]
    with pytest.raises((TypeError, ValueError), match="[Nn]ormalization|NORMALIZATION"):
        package.bind_runtime_defaults({selector.lower(): "invalid"})


@pytest.mark.parametrize("catalog_key", _PACKAGES)
def test_residual_stack_selects_its_own_normalization(catalog_key):
    package = model_package(catalog_key)
    supported = {key.lower() for key in package.runtime_defaults_spec.supported_keys}
    residual_prefix = (
        "stack" if "stack_residual_model_flag" in supported else "ff_stack"
    )
    overrides = _small_overrides(package)
    overrides.update(
        {key.lower(): NormalizationOptions.RMS_NORM for key in _selectors(package)}
    )
    overrides.update(
        {
            f"{residual_prefix}_residual_connection_option": WeightedResidualConfig,
            f"{residual_prefix}_residual_model_flag": True,
            "residual_stack_independent_flag": True,
            "residual_stack_normalization": NormalizationOptions.DYNAMIC_TANH,
            "residual_stack_layer_norm_position": LayerNormPositionOptions.BEFORE,
            "residual_stack_num_layers": 2,
            "residual_stack_hidden_dim": 8,
        }
    )
    configuration = package.build_configuration(config_overrides=overrides)
    model = package.build_model(configuration)
    selected = [
        name
        for name, module in model.named_modules()
        if type(module).__name__ == "DynamicTanh"
    ]
    assert selected
    assert all("residual" in name for name in selected), selected
    outputs = _forward(package, model)
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    assert all(torch.isfinite(tensor).all() for tensor in tensors)


def test_controller_inherits_parent_type_when_position_is_overridden():
    package = model_package("linears/linear")
    configuration = package.build_configuration(
        config_overrides={
            **_small_overrides(package),
            "normalization": NormalizationOptions.RMS_NORM,
            "submodule_stack_normalization": NormalizationOptions.DERF,
            "stack_halting_flag": True,
            "halting_stack_independent_flag": True,
            "halting_stack_layer_norm_position": LayerNormPositionOptions.BEFORE,
            "halting_stack_num_layers": 2,
            "halting_stack_hidden_dim": 8,
        }
    )
    model = package.build_model(configuration)
    controller_norms = [
        module
        for name, module in model.named_modules()
        if "halting" in name and type(module).__name__ in _NORMALIZATION_TYPES.values()
    ]
    assert controller_norms
    assert all(type(module).__name__ == "DynamicErf" for module in controller_norms)
    outputs = _forward(package, model)
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    assert all(torch.isfinite(tensor).all() for tensor in tensors)


@pytest.mark.parametrize(
    "catalog_key", tuple(k for k in _PACKAGES if k.startswith("transformer/"))
)
def test_encoder_decoder_and_output_norms_select_independently(catalog_key):
    package = model_package(catalog_key)
    configuration = package.build_configuration(
        config_overrides={
            **_small_overrides(package),
            "encoder_normalization": NormalizationOptions.DYNAMIC_TANH,
            "decoder_normalization": NormalizationOptions.DERF,
            "encoder_output_normalization": NormalizationOptions.LAYER_NORM,
            "decoder_output_normalization": NormalizationOptions.RMS_NORM,
        }
    )
    model = package.build_model(configuration)
    for stack, expected in (
        (model.encoder, "DynamicTanh"),
        (model.decoder, "DynamicErf"),
    ):
        norms = [
            m
            for m in stack.modules()
            if type(m).__name__ in _NORMALIZATION_TYPES.values()
        ]
        assert norms
        assert all(type(m).__name__ == expected for m in norms)
    assert type(model.encoder_layer_norm).__name__ == "LayerNorm"
    assert type(model.decoder_layer_norm).__name__ == "RMSNorm"
    assert all(torch.isfinite(tensor).all() for tensor in _forward(package, model))


@pytest.mark.parametrize(
    "catalog_key", tuple(k for k in _PACKAGES if k.startswith("bert/"))
)
def test_embedding_encoder_and_mlm_norms_select_independently(catalog_key):
    package = model_package(catalog_key)
    configuration = package.build_configuration(
        config_overrides={
            **_small_overrides(package),
            "normalization": NormalizationOptions.RMS_NORM,
            "embedding_normalization": NormalizationOptions.DYNAMIC_TANH,
            "mlm_normalization": NormalizationOptions.DERF,
            "encoder_output_normalization": NormalizationOptions.DYISRU,
            "layer_norm_position": LayerNormPositionOptions.BEFORE,
        }
    )
    model = package.build_model(configuration)
    assert type(model.embedding_layer_norm).__name__ == "DynamicTanh"
    assert type(model.mlm_layer_norm).__name__ == "DynamicErf"
    assert type(model.encoder_layer_norm).__name__ == "DynamicISRU"
    assert all(torch.isfinite(tensor).all() for tensor in _forward(package, model))
