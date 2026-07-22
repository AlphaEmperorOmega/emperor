from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from model_runtime.packages import (
    MODEL_ID_SEGMENT_RE,
    ModelIdentity,
    ModelPackage,
    is_safe_model_identity,
    is_safe_model_segment,
    model_key,
    split_model_id,
)
from models.bert.expert_linear import MODEL_PACKAGE as BERT_EXPERT_LINEAR
from models.bert.expert_linear_adaptive import (
    MODEL_PACKAGE as BERT_EXPERT_LINEAR_ADAPTIVE,
)
from models.bert.linear import MODEL_PACKAGE as BERT_LINEAR
from models.bert.linear_adaptive import MODEL_PACKAGE as BERT_LINEAR_ADAPTIVE
from models.experts.linear import MODEL_PACKAGE as EXPERTS_LINEAR
from models.experts.linear_adaptive import MODEL_PACKAGE as EXPERTS_LINEAR_ADAPTIVE
from models.gpt.expert_linear import MODEL_PACKAGE as GPT_EXPERT_LINEAR
from models.gpt.expert_linear_adaptive import (
    MODEL_PACKAGE as GPT_EXPERT_LINEAR_ADAPTIVE,
)
from models.gpt.linear import MODEL_PACKAGE as GPT_LINEAR
from models.gpt.linear_adaptive import MODEL_PACKAGE as GPT_LINEAR_ADAPTIVE
from models.linears.linear import MODEL_PACKAGE as LINEARS_LINEAR
from models.linears.linear_adaptive import MODEL_PACKAGE as LINEARS_LINEAR_ADAPTIVE
from models.neuron.expert_linear import MODEL_PACKAGE as NEURON_EXPERT_LINEAR
from models.neuron.expert_linear_adaptive import (
    MODEL_PACKAGE as NEURON_EXPERT_LINEAR_ADAPTIVE,
)
from models.neuron.linear import MODEL_PACKAGE as NEURON_LINEAR
from models.neuron.linear_adaptive import MODEL_PACKAGE as NEURON_LINEAR_ADAPTIVE
from models.parametric.parametric_generator import (
    MODEL_PACKAGE as PARAMETRIC_GENERATOR,
)
from models.parametric.parametric_matrix import MODEL_PACKAGE as PARAMETRIC_MATRIX
from models.parametric.parametric_vector import MODEL_PACKAGE as PARAMETRIC_VECTOR
from models.transformer.expert_linear import (
    MODEL_PACKAGE as TRANSFORMER_EXPERT_LINEAR,
)
from models.transformer.expert_linear_adaptive import (
    MODEL_PACKAGE as TRANSFORMER_EXPERT_LINEAR_ADAPTIVE,
)
from models.transformer.linear import MODEL_PACKAGE as TRANSFORMER_LINEAR
from models.transformer.linear_adaptive import (
    MODEL_PACKAGE as TRANSFORMER_LINEAR_ADAPTIVE,
)
from models.vit.expert_linear import MODEL_PACKAGE as VIT_EXPERT_LINEAR
from models.vit.expert_linear_adaptive import (
    MODEL_PACKAGE as VIT_EXPERT_LINEAR_ADAPTIVE,
)
from models.vit.linear import MODEL_PACKAGE as VIT_LINEAR
from models.vit.linear_adaptive import MODEL_PACKAGE as VIT_LINEAR_ADAPTIVE

_PACKAGES = (
    BERT_LINEAR,
    BERT_LINEAR_ADAPTIVE,
    BERT_EXPERT_LINEAR,
    BERT_EXPERT_LINEAR_ADAPTIVE,
    GPT_LINEAR,
    GPT_LINEAR_ADAPTIVE,
    GPT_EXPERT_LINEAR,
    GPT_EXPERT_LINEAR_ADAPTIVE,
    VIT_LINEAR,
    VIT_LINEAR_ADAPTIVE,
    VIT_EXPERT_LINEAR,
    VIT_EXPERT_LINEAR_ADAPTIVE,
    TRANSFORMER_LINEAR,
    TRANSFORMER_LINEAR_ADAPTIVE,
    TRANSFORMER_EXPERT_LINEAR,
    TRANSFORMER_EXPERT_LINEAR_ADAPTIVE,
    LINEARS_LINEAR,
    LINEARS_LINEAR_ADAPTIVE,
    EXPERTS_LINEAR,
    EXPERTS_LINEAR_ADAPTIVE,
    PARAMETRIC_VECTOR,
    PARAMETRIC_MATRIX,
    PARAMETRIC_GENERATOR,
    NEURON_LINEAR,
    NEURON_LINEAR_ADAPTIVE,
    NEURON_EXPERT_LINEAR,
    NEURON_EXPERT_LINEAR_ADAPTIVE,
)

MODEL_CATALOG: dict[str, ModelPackage] = {
    package.identity.catalog_key: package for package in _PACKAGES
}

MODEL_ORDER: dict[str, int] = {
    "bert/linear": 0,
    "bert/linear_adaptive": 1,
    "bert/expert_linear": 2,
    "bert/expert_linear_adaptive": 3,
    "gpt/linear": 0,
    "gpt/linear_adaptive": 1,
    "gpt/expert_linear": 2,
    "gpt/expert_linear_adaptive": 3,
    "vit/linear": 0,
    "vit/linear_adaptive": 1,
    "vit/expert_linear": 2,
    "vit/expert_linear_adaptive": 3,
    "transformer/linear": 0,
    "transformer/linear_adaptive": 1,
    "transformer/expert_linear": 2,
    "transformer/expert_linear_adaptive": 3,
    "neuron/linear": 0,
    "neuron/linear_adaptive": 1,
    "neuron/expert_linear": 2,
    "neuron/expert_linear_adaptive": 3,
}

EMPTY_CATEGORY_PACKAGES: set[str] = set()


def is_safe_model_id(model_id: object) -> bool:
    return split_model_id(model_id) is not None


def model_package(catalog_key: str) -> ModelPackage | None:
    identity = split_model_id(catalog_key)
    if identity is None:
        return None
    return MODEL_CATALOG.get(identity.catalog_key)


def model_identity_for_parts(model_type: str, model: str) -> ModelIdentity | None:
    if not is_safe_model_identity(model_type, model):
        return None
    package = MODEL_CATALOG.get(model_key(model_type, model))
    return package.identity if package is not None else None


def model_id_from_parts(model_type: str, model: str) -> str | None:
    identity = model_identity_for_parts(model_type, model)
    return identity.catalog_key if identity is not None else None


def model_id_from_payload(payload: Mapping[str, Any]) -> str | None:
    model_type = payload.get("modelType")
    model = payload.get("model")
    if not isinstance(model_type, str) or not isinstance(model, str):
        return None
    return model_id_from_parts(model_type, model)


def model_identity_payload(catalog_key: str) -> dict[str, str]:
    package = model_package(catalog_key)
    if package is None:
        raise ValueError(f"Unknown Model Package identity: {catalog_key!r}")
    return package.identity.to_payload()


def discover_model_ids() -> list[str]:
    return sorted(
        MODEL_CATALOG,
        key=lambda key: (key.split("/", 1)[0], MODEL_ORDER.get(key, key)),
    )


def discover_model_packages() -> list[ModelPackage]:
    return [MODEL_CATALOG[key] for key in discover_model_ids()]


def discover_model_types() -> list[str]:
    return sorted({package.identity.model_type for package in _PACKAGES})


def model_type_exists(model_type: str) -> bool:
    return is_safe_model_segment(model_type) and model_type in discover_model_types()


def discover_model_identities() -> list[ModelIdentity]:
    return [package.identity for package in discover_model_packages()]


def discover_model_identities_for_type(model_type: str) -> list[ModelIdentity]:
    if not model_type_exists(model_type):
        return []
    return [
        identity
        for identity in discover_model_identities()
        if identity.model_type == model_type
    ]


def discover_model_identity_payloads() -> list[dict[str, str]]:
    return [identity.to_payload() for identity in discover_model_identities()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List canonical Emperor Model Package identities."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--list-types", action="store_true")
    parser.add_argument("--model-type")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_types:
        for model_type in discover_model_types():
            print(model_type)
        return
    if args.model_type:
        if not model_type_exists(args.model_type):
            raise SystemExit(f"Unknown model type: --model-type {args.model_type}")
        for identity in discover_model_identities_for_type(args.model_type):
            print(identity.catalog_key)
        return
    for identity in discover_model_identities():
        print(identity.catalog_key)


__all__ = [
    "EMPTY_CATEGORY_PACKAGES",
    "MODEL_CATALOG",
    "MODEL_ID_SEGMENT_RE",
    "MODEL_ORDER",
    "ModelIdentity",
    "ModelPackage",
    "discover_model_identities",
    "discover_model_identities_for_type",
    "discover_model_identity_payloads",
    "discover_model_ids",
    "discover_model_packages",
    "discover_model_types",
    "is_safe_model_id",
    "is_safe_model_identity",
    "is_safe_model_segment",
    "model_id_from_parts",
    "model_id_from_payload",
    "model_identity_for_parts",
    "model_identity_payload",
    "model_key",
    "model_package",
    "model_type_exists",
    "split_model_id",
]


if __name__ == "__main__":
    main()
