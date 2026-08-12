from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from emperor.layers import RecurrentLayerConfig
from models.catalog import model_package

_PACKAGE_CASES = (
    "bert/expert_linear_adaptive",
    "gpt/expert_linear_adaptive",
)


def _core_config(configuration):
    experiment_config = configuration.experiment_config
    encoder_config = getattr(experiment_config, "encoder_config", None)
    if encoder_config is not None:
        return encoder_config
    return experiment_config.decoder_config


def _core_layer_config(core_config):
    stack_config = (
        core_config.block_config
        if isinstance(core_config, RecurrentLayerConfig)
        else core_config
    )
    return stack_config.layer_config.layer_model_config


def _attention_config(core_layer_config):
    attention_config = getattr(core_layer_config, "attention_config", None)
    if attention_config is not None:
        return attention_config
    return core_layer_config.self_attention_config


@pytest.mark.parametrize("catalog_key", _PACKAGE_CASES, ids=("bert", "gpt"))
def test_recurrent_flat_values_stay_scoped_without_mutating_the_input(
    catalog_key: str,
) -> None:
    overrides = {
        "recurrent_flag": True,
        "recurrent_max_steps": 7,
        "expert_recurrent_flag": True,
        "expert_recurrent_max_steps": 11,
    }
    original = dict(overrides)
    package = model_package(catalog_key)
    assert package is not None

    runtime = package.bind_runtime_defaults(overrides)
    configuration = package.build_configuration(config_overrides=overrides)

    assert overrides == original
    assert type(runtime) is package.runtime_options_type
    core_config = _core_config(configuration)
    assert isinstance(core_config, RecurrentLayerConfig)
    assert core_config.max_steps == 7

    core_layer_config = _core_layer_config(core_config)
    attention_expert = _attention_config(
        core_layer_config
    ).experts_config.expert_model_config
    feed_forward_mixture = core_layer_config.feed_forward_config.stack_config
    feed_forward_expert = feed_forward_mixture.stack_config.layer_config.layer_model_config.expert_model_config
    for expert_config in (attention_expert, feed_forward_expert):
        assert isinstance(expert_config, RecurrentLayerConfig)
        assert expert_config.max_steps == 11


@pytest.mark.parametrize("catalog_key", _PACKAGE_CASES, ids=("bert", "gpt"))
def test_control_groups_broadcast_to_the_outer_mixture_without_leaking_to_experts(
    catalog_key: str,
) -> None:
    overrides = {
        "stack_gate_flag": True,
        "gate_stack_independent_flag": True,
        "gate_stack_hidden_dim": 37,
        "memory_flag": True,
        "memory_stack_independent_flag": True,
        "memory_stack_num_layers": 4,
    }
    original = dict(overrides)
    package = model_package(catalog_key)
    assert package is not None

    package.bind_runtime_defaults(overrides)
    configuration = package.build_configuration(config_overrides=overrides)

    assert overrides == original
    core_config = _core_config(configuration)
    core_stack = (
        core_config.block_config
        if isinstance(core_config, RecurrentLayerConfig)
        else core_config
    )
    assert core_stack.layer_config.gate_config.model_config.hidden_dim == 37
    assert core_stack.shared_memory_config.model_config.num_layers == 4

    core_layer_config = core_stack.layer_config.layer_model_config
    attention_expert = _attention_config(
        core_layer_config
    ).experts_config.expert_model_config
    feed_forward_mixture = core_layer_config.feed_forward_config.stack_config
    mixture_stack = feed_forward_mixture.stack_config
    assert mixture_stack.layer_config.gate_config.model_config.hidden_dim == 37
    assert mixture_stack.shared_memory_config.model_config.num_layers == 4

    feed_forward_expert = (
        mixture_stack.layer_config.layer_model_config.expert_model_config
    )
    for expert_stack in (attention_expert, feed_forward_expert):
        assert expert_stack.layer_config.gate_config is None
        assert expert_stack.shared_memory_config is None


@pytest.mark.parametrize("catalog_key", _PACKAGE_CASES, ids=("bert", "gpt"))
def test_public_runtime_interface_preserves_exact_unknown_key_error(
    catalog_key: str,
) -> None:
    package = model_package(catalog_key)
    assert package is not None

    with pytest.raises(ValueError) as error:
        package.bind_runtime_defaults({"unknown_contract_key": 1})

    package_name = "models." + catalog_key.replace("/", ".")
    assert str(error.value) == (
        f"{package_name}: unknown Runtime Defaults field(s): 'unknown_contract_key'"
    )


@pytest.mark.parametrize("catalog_key", _PACKAGE_CASES, ids=("bert", "gpt"))
@pytest.mark.parametrize(
    "module_order",
    (
        (
            "runtime_defaults",
            "runtime_options",
            "config_builder",
            "presets",
        ),
        (
            "presets",
            "config_builder",
            "runtime_options",
            "runtime_defaults",
        ),
    ),
    ids=("runtime-first", "construction-first"),
)
def test_public_module_import_orders_are_cycle_free(
    catalog_key: str,
    module_order: tuple[str, ...],
) -> None:
    source_root = Path(__file__).parents[2] / "src"
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/src-models-mpl"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH")))
    )
    package_name = "models." + catalog_key.replace("/", ".")
    imports = "; ".join(
        f"import {package_name}.{module_name}" for module_name in module_order
    )

    subprocess.run(
        [sys.executable, "-c", imports],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
