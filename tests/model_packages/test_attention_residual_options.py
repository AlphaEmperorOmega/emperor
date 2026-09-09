from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import pytest
import torch

from emperor.layers import (
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LayerNormPositionOptions,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from model_runtime.inspection import configuration_schema
from models.catalog import discover_model_packages, model_package

PACKAGES = tuple(package.catalog_key for package in discover_model_packages())
SETTINGS = (("RESIDUAL_BLOCK_SIZE", int, 3), ("RESIDUAL_RMS_NORM_EPSILON", float, 2e-5))


def walk(value):
    pending = [value]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if is_dataclass(current):
            pending.extend(getattr(current, field.name) for field in fields(current))
        elif isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
        elif hasattr(current, "_values"):
            pending.append(current._values)


def small_overrides(package):
    family = package.identity.model_type
    proposed = {
        "batch_size": 2,
        "hidden_dim": 8,
        "stack_num_layers": 2,
        "stack_dropout_probability": 0.0,
        "residual_stack_independent_flag": True,
        "residual_stack_hidden_dim": 8,
        "residual_stack_num_layers": 2,
        "residual_stack_dropout_probability": 0.0,
        "residual_stack_layer_norm_position": LayerNormPositionOptions.DISABLED,
        "residual_stack_apply_output_postprocessing_flag": True,
        "residual_stack_residual_connection_option": AdditiveResidualConfig,
    }
    if family in ("bert", "gpt", "vit", "transformer"):
        prefix = "attn_stack"
        proposed.update(
            attn_num_heads=2,
            attn_stack_num_layers=2,
            attn_stack_hidden_dim=8,
            attn_stack_apply_output_postprocessing_flag=True,
            attn_stack_dropout_probability=0.0,
            ff_stack_hidden_dim=8,
        )
        if family in ("bert", "gpt"):
            proposed.update(input_dim=32, output_dim=32, sequence_length=4)
        elif family == "transformer":
            proposed.update(
                vocab_size=32,
                model_dim=8,
                source_sequence_length=4,
                target_sequence_length=4,
                encoder_num_layers=1,
                decoder_num_layers=1,
                dropout_probability=0.0,
            )
    else:
        prefix = "stack"
    supported = {key.lower() for key in package.runtime_defaults_spec.supported_keys}
    return prefix, {key: value for key, value in proposed.items() if key in supported}


def synthetic_input(package):
    if package.identity.model_type in ("bert", "gpt", "transformer"):
        tokens = torch.tensor([[2, 7, 8, 3], [2, 9, 10, 3]])
        return (
            (tokens, tokens)
            if package.identity.model_type == "transformer"
            else (tokens,)
        )
    dataset = package.resolve_dataset(None)
    return (
        torch.randn(
            2, dataset.num_channels, dataset.default_height, dataset.default_width
        ),
    )


def tensors(output):
    return output if isinstance(output, tuple) else (output,)


@pytest.mark.parametrize("catalog_key", PACKAGES)
def test_every_residual_selector_exposes_explicit_attention_settings(catalog_key):
    package = model_package(catalog_key)
    spec = package.runtime_defaults_spec
    schema = {field.key: field for field in configuration_schema(package).fields}
    selectors = [
        key for key in spec.supported_keys if key.endswith("RESIDUAL_CONNECTION_OPTION")
    ]
    assert selectors
    for selector in selectors:
        for suffix, field_type, _ in SETTINGS:
            key = selector.replace("RESIDUAL_CONNECTION_OPTION", suffix)
            assert key in spec.supported_keys, key
            assert spec.annotations[key] == field_type | None, key
            assert spec.current_value(key) is None, key
            assert (key in schema) == (selector in schema), key
            assert (
                spec.configuration_metadata[key]["sectionPath"]
                == spec.configuration_metadata[selector]["sectionPath"]
            )
            if key in schema:
                assert "Supply explicitly" in schema[key].description
                assert (
                    "1e-6" in schema[key].description
                    if field_type is float
                    else "1 for full attention" in schema[key].description
                )


@pytest.mark.parametrize("catalog_key", PACKAGES)
def test_attention_settings_reach_direct_runtime_options(catalog_key):
    package = model_package(catalog_key)
    spec = package.runtime_defaults_spec
    selectors = sorted(
        key for key in spec.supported_keys if key.endswith("RESIDUAL_CONNECTION_OPTION")
    )
    for selector in selectors:
        prefix = selector.removesuffix("RESIDUAL_CONNECTION_OPTION")
        overrides = {
            prefix.lower() + suffix.lower(): value for suffix, _, value in SETTINGS
        }
        independent_key = prefix + "INDEPENDENT_FLAG"
        if independent_key in spec.supported_keys:
            overrides[independent_key.lower()] = True
        runtime = package.bind_runtime_defaults(overrides)
        found = {suffix: [] for suffix, _, _ in SETTINGS}
        for value in walk(runtime):
            items = (
                ((field.name, getattr(value, field.name)) for field in fields(value))
                if is_dataclass(value)
                else value.items()
                if isinstance(value, Mapping)
                else ()
            )
            for key, field_value in items:
                for suffix, _, _ in SETTINGS:
                    if str(key).lower().endswith(suffix.lower()):
                        found[suffix].append(field_value)
        for suffix, _, expected in SETTINGS:
            assert expected in found[suffix], (catalog_key, selector, suffix)


MODES = [
    (AdditiveResidualConfig, False, 1),
    (WeightedResidualConfig, False, 1),
    (WeightedResidualConfig, True, 1),
    (WeightedBlendResidualConfig, False, 1),
    (WeightedBlendResidualConfig, True, 1),
    (AttentionResidualConfig, False, 1),
    (AttentionResidualConfig, True, 1),
    (AttentionResidualConfig, False, 2),
    (AttentionResidualConfig, True, 2),
]


def assert_model_residual_options(
    catalog_key, selector, modeled, block_size, *, placement=None
):
    torch.manual_seed(31)
    package = model_package(catalog_key)
    prefix, overrides = small_overrides(package)
    if placement is not None:
        prefix = placement
        if placement == "recurrent":
            overrides.update(
                recurrent_flag=True,
                recurrent_max_steps=2,
                recurrent_initial_iterations=2,
            )
    overrides.update(
        {
            f"{prefix}_residual_connection_option": selector,
            f"{prefix}_residual_model_flag": modeled,
            f"{prefix}_residual_block_size": block_size,
            f"{prefix}_residual_rms_norm_epsilon": 2e-5,
        }
    )
    configuration = package.build_configuration(config_overrides=overrides)
    model = package.build_model(configuration).train()
    residuals = [
        module
        for module in model.modules()
        if type(getattr(module, "cfg", None)) is selector
    ]
    assert residuals, (catalog_key, prefix)
    inputs = synthetic_input(package)
    visited = set()
    hooks = [
        module.register_forward_hook(lambda module, args, output: visited.add(module))
        for module in residuals
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    try:
        outputs = tensors(model(*inputs))
    finally:
        for hook in hooks:
            hook.remove()
    assert visited, (catalog_key, prefix)
    assert all(torch.isfinite(output).all() for output in outputs)
    sum(output.float().square().mean() for output in outputs).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    updated = max(
        (p for p in model.parameters() if p.grad is not None),
        key=lambda p: p.grad.abs().max().item(),
    )
    before = updated.detach().clone()
    optimizer.step()
    assert not torch.equal(before, updated)
    if selector is AttentionResidualConfig:
        assert all(
            type(residual) is residual.cfg.registry_owner() for residual in residuals
        )
        for residual in residuals:
            assert residual.block_size == block_size
            assert residual.rms_norm_epsilon == 2e-5
            assert not hasattr(residual, "model")
            if modeled:
                assert residual.query is None
                assert residual.query_model.input_dim == residual.residual_dim
                assert residual.query_model.output_dim == residual.residual_dim
                assert any(
                    key.startswith("query_model.") for key in residual.state_dict()
                )
            else:
                assert residual.query_model is None
                assert tuple(residual.state_dict()) == ("query", "key_norm.weight")
        if modeled:
            assert len({id(residual.query_model) for residual in residuals}) == len(
                residuals
            )
            assert any(
                p.grad is not None and torch.count_nonzero(p.grad)
                for residual in visited
                for p in residual.query_model.parameters()
            )
    elif modeled:
        assert all(
            residual.model.input_dim == 2 * residual.residual_dim
            and residual.model.output_dim == residual.residual_dim
            for residual in residuals
        )
    model.eval()
    expected = tensors(model(*inputs))
    repeated = tensors(model(*inputs))
    for actual, previous in zip(repeated, expected, strict=True):
        torch.testing.assert_close(actual, previous)
    restored = package.build_model(configuration).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    for actual, previous in zip(tensors(restored(*inputs)), expected, strict=True):
        torch.testing.assert_close(actual, previous)


@pytest.mark.parametrize("catalog_key", PACKAGES)
@pytest.mark.parametrize("selector,modeled,block_size", MODES)
def test_model_residual_options_forward_update_and_checkpoint(
    catalog_key, selector, modeled, block_size
):
    assert_model_residual_options(catalog_key, selector, modeled, block_size)


ADDITIONAL_PLACEMENTS = tuple(
    (package.catalog_key, prefix)
    for package in discover_model_packages()
    for prefix in ("stack", "recurrent")
    if prefix != small_overrides(package)[0]
    and f"{prefix.upper()}_RESIDUAL_CONNECTION_OPTION"
    in package.runtime_defaults_spec.supported_keys
)


@pytest.mark.parametrize("catalog_key,placement", ADDITIONAL_PLACEMENTS)
@pytest.mark.parametrize(
    "modeled,block_size", [(False, 1), (True, 1), (False, 2), (True, 2)]
)
def test_main_and_recurrent_attention_placements_forward_update_and_checkpoint(
    catalog_key, placement, modeled, block_size
):
    assert_model_residual_options(
        catalog_key, AttentionResidualConfig, modeled, block_size, placement=placement
    )




@pytest.mark.parametrize(
    "catalog_key",
    [
        key
        for key in PACKAGES
        if key.split("/")[0] in ("bert", "gpt", "vit", "mlp_mixer", "transformer")
    ],
)
def test_native_transformer_joins_reject_attention_without_shared_history(catalog_key):
    package = model_package(catalog_key)
    _, overrides = small_overrides(package)
    configuration = package.build_configuration(config_overrides=overrides)
    from emperor.transformer import (
        TransformerDecoderLayerConfig,
        TransformerEncoderLayerConfig,
    )

    native_layers = [
        value
        for value in walk(configuration)
        if isinstance(
            value, (TransformerEncoderLayerConfig, TransformerDecoderLayerConfig)
        )
    ]
    assert native_layers
    for native_layer in native_layers:
        native_layer.residual_config = AttentionResidualConfig(
            block_size=1,
            rms_norm_epsilon=1e-6,
        )
    with pytest.raises(
        ValueError, match="share an explicit forward-local history bridge"
    ):
        package.build_model(configuration)


@pytest.mark.parametrize(
    "catalog_key", [key for key in PACKAGES if key.startswith("mlp_mixer/")]
)
def test_mixer_join_selector_rejects_attention_without_shared_history(catalog_key):
    package = model_package(catalog_key)
    _, overrides = small_overrides(package)
    overrides.update(
        {
            "mixer_residual_connection_option": AttentionResidualConfig,
            "mixer_residual_model_flag": True,
            "mixer_residual_block_size": 1,
            "mixer_residual_rms_norm_epsilon": 1e-6,
        }
    )
    configuration = package.build_configuration(config_overrides=overrides)
    with pytest.raises(
        ValueError, match="share an explicit forward-local history bridge"
    ):
        package.build_model(configuration)
