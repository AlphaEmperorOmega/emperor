from __future__ import annotations

from models.catalog import discover_model_packages


def _direct_and_recurrent(role: str) -> set[str]:
    return {f"{role}HALTING_OPTION", f"{role}RECURRENT_HALTING_OPTION"}


def _expected_halting_options(catalog_key: str) -> set[str]:
    family, variant = catalog_key.split("/", 1)
    if family == "parametric":
        return set()
    if family == "linears":
        return _direct_and_recurrent("")
    if family == "experts":
        expected = _direct_and_recurrent("") | _direct_and_recurrent("EXPERT_")
        if variant == "linear_adaptive":
            expected |= _direct_and_recurrent("ROUTER_")
        return expected
    if family == "neuron":
        expected = _direct_and_recurrent("") | {"CLUSTER_HALTING_OPTION"}
        if variant.startswith("expert_"):
            expected |= _direct_and_recurrent("EXPERT_")
        if variant == "expert_linear_adaptive":
            expected |= _direct_and_recurrent("ROUTER_")
        return expected
    if family == "mlp_mixer":
        expected = (
            _direct_and_recurrent("")
            | _direct_and_recurrent("TOKEN_MIXER_")
            | _direct_and_recurrent("CHANNEL_MIXER_")
        )
        if variant.startswith("expert_"):
            expected |= _direct_and_recurrent("EXPERT_")
        return expected
    if family == "transformer":
        expected = _direct_and_recurrent("ATTN_") | _direct_and_recurrent("FF_")
        for role in (
            "ENCODER_ATTN_",
            "ENCODER_FF_",
            "DECODER_SELF_ATTN_",
            "DECODER_CROSS_ATTN_",
            "DECODER_FF_",
        ):
            expected |= _direct_and_recurrent(role)
        if variant.startswith("expert_"):
            expected |= _direct_and_recurrent("EXPERT_")
            expected |= _direct_and_recurrent("ROUTER_")
        return expected

    expected = (
        _direct_and_recurrent("")
        | _direct_and_recurrent("ATTN_")
        | _direct_and_recurrent("FF_")
    )
    if variant.startswith("expert_"):
        expected |= _direct_and_recurrent("EXPERT_")
    if variant == "expert_linear_adaptive":
        expected |= _direct_and_recurrent("ROUTER_")
    return expected


def test_every_package_advertises_its_complete_halting_strategy_surface() -> None:
    for package in discover_model_packages():
        advertised = {
            key
            for key in package.runtime_defaults_spec.supported_keys
            if key.endswith("HALTING_OPTION")
        }
        assert advertised == _expected_halting_options(package.catalog_key)
