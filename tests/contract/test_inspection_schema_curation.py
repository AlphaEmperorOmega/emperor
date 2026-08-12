from __future__ import annotations

from model_runtime.inspection import configuration_schema
from models.catalog import discover_model_packages


def test_schema_skip_keys_are_presentation_only_and_remain_cli_reachable() -> None:
    curated_packages: set[str] = set()
    for package in discover_model_packages():
        spec = package.runtime_defaults_spec
        skipped = spec.skipped_schema_keys
        if not skipped:
            continue
        curated_packages.add(package.catalog_key)
        supported = set(spec.supported_keys)
        schema_keys = {field.key for field in configuration_schema(package).fields}

        assert skipped <= supported
        assert skipped.isdisjoint(schema_keys)

    assert curated_packages == {
        "bert/linear_adaptive",
        "bert/expert_linear",
        "gpt/linear_adaptive",
        "gpt/expert_linear",
        "vit/linear",
        "vit/linear_adaptive",
        "vit/expert_linear",
        "vit/expert_linear_adaptive",
    }
