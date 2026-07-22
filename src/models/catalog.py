from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from model_runtime.packages.definition import ModelCatalogEntry, ModelPackage
from model_runtime.packages.identity import (
    MODEL_ID_SEGMENT_RE,
    ModelIdentity,
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
from models.transformer.expert_linear import MODEL_PACKAGE as TRANSFORMER_EXPERT_LINEAR
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

MODEL_CATALOG: dict[str, ModelPackage] = {
    "bert/linear": BERT_LINEAR,
    "bert/linear_adaptive": BERT_LINEAR_ADAPTIVE,
    "bert/expert_linear": BERT_EXPERT_LINEAR,
    "bert/expert_linear_adaptive": BERT_EXPERT_LINEAR_ADAPTIVE,
    "gpt/linear": GPT_LINEAR,
    "gpt/linear_adaptive": GPT_LINEAR_ADAPTIVE,
    "gpt/expert_linear": GPT_EXPERT_LINEAR,
    "gpt/expert_linear_adaptive": GPT_EXPERT_LINEAR_ADAPTIVE,
    "vit/linear": VIT_LINEAR,
    "vit/linear_adaptive": VIT_LINEAR_ADAPTIVE,
    "vit/expert_linear": VIT_EXPERT_LINEAR,
    "vit/expert_linear_adaptive": VIT_EXPERT_LINEAR_ADAPTIVE,
    "transformer/linear": TRANSFORMER_LINEAR,
    "transformer/linear_adaptive": TRANSFORMER_LINEAR_ADAPTIVE,
    "transformer/expert_linear": TRANSFORMER_EXPERT_LINEAR,
    "transformer/expert_linear_adaptive": TRANSFORMER_EXPERT_LINEAR_ADAPTIVE,
    "linears/linear": LINEARS_LINEAR,
    "linears/linear_adaptive": LINEARS_LINEAR_ADAPTIVE,
    "experts/linear": EXPERTS_LINEAR,
    "experts/linear_adaptive": EXPERTS_LINEAR_ADAPTIVE,
    "parametric/parametric_vector": PARAMETRIC_VECTOR,
    "parametric/parametric_matrix": PARAMETRIC_MATRIX,
    "parametric/parametric_generator": PARAMETRIC_GENERATOR,
    "neuron/linear": NEURON_LINEAR,
    "neuron/linear_adaptive": NEURON_LINEAR_ADAPTIVE,
    "neuron/expert_linear": NEURON_EXPERT_LINEAR,
    "neuron/expert_linear_adaptive": NEURON_EXPERT_LINEAR_ADAPTIVE,
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


def _module_path_for_package(package: ModelPackage) -> str:
    return package.module_path or (
        f"models.{package.identity.model_type}.{package.identity.model}"
    )


_MODULE_TO_PUBLIC_ID = {
    _module_path_for_package(package): public_id
    for public_id, package in MODEL_CATALOG.items()
}


def _flat_name_priority(public_id: str) -> int:
    if public_id.startswith("linears/") or public_id.startswith("experts/"):
        return 0
    if public_id.startswith("bert/"):
        return 1
    if public_id.startswith("vit/"):
        return 2
    if public_id.startswith("neuron/"):
        return 3
    return 4


FLAT_TO_PUBLIC_ID: dict[str, str] = {}
for public_id in sorted(MODEL_CATALOG, key=_flat_name_priority):
    flat_name = public_id.rsplit("/", 1)[-1]
    FLAT_TO_PUBLIC_ID.setdefault(flat_name, public_id)


def is_safe_model_id(model_id: object) -> bool:
    if not isinstance(model_id, str):
        return False
    if not model_id or model_id.strip() != model_id or "\\" in model_id:
        return False
    segments = model_id.split("/")
    if len(segments) < 2:
        return False
    return all(
        segment not in {".", ".."} and MODEL_ID_SEGMENT_RE.fullmatch(segment)
        for segment in segments
    )


def model_identity_payload(model_id: str) -> dict[str, str]:
    identity = model_identity_for_model_id(model_id)
    if identity is None:
        identity = split_model_id(model_id)
        if identity is None:
            raise ValueError(f"Invalid model id: {model_id}")
    return identity.to_payload()


def model_identity_for_model_id(model_id: str) -> ModelIdentity | None:
    package = model_package(model_id)
    return package.identity if package is not None else None


def model_identity_for_parts(model_type: str, model: str) -> ModelIdentity | None:
    if not is_safe_model_identity(model_type, model):
        return None
    package = MODEL_CATALOG.get(model_key(model_type, model))
    return package.identity if package is not None else None


def model_id_from_parts(model_type: str, model: str) -> str | None:
    identity = model_identity_for_parts(model_type, model)
    return identity.catalog_key if identity is not None else None


def discover_model_ids() -> list[str]:
    return sorted(
        MODEL_CATALOG,
        key=lambda key: (key.split("/", 1)[0], MODEL_ORDER.get(key, key)),
    )


def discover_model_packages() -> list[ModelPackage]:
    return [MODEL_CATALOG[key] for key in discover_model_ids()]


def discover_model_types() -> list[str]:
    return sorted({package.model_type for package in MODEL_CATALOG.values()})


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


def model_package(model_id: str) -> ModelPackage | None:
    if not is_safe_model_id(model_id):
        return None
    return MODEL_CATALOG.get(model_id)


def catalog_entry(model_id: str) -> ModelPackage | None:
    return model_package(model_id)


def module_path_for_model_id(model_id: str) -> str | None:
    package = model_package(model_id)
    return _module_path_for_package(package) if package is not None else None


def module_path_for_model_identity(model_type: str, model: str) -> str | None:
    model_id = model_id_from_parts(model_type, model)
    return module_path_for_model_id(model_id) if model_id is not None else None


def public_id_for_module(module_path: str) -> str | None:
    if module_path in _MODULE_TO_PUBLIC_ID:
        return _MODULE_TO_PUBLIC_ID[module_path]
    for package, public_id in _MODULE_TO_PUBLIC_ID.items():
        if module_path.startswith(f"{package}."):
            return public_id
    return None


def model_package_for_module(module_path: str) -> ModelPackage | None:
    public_id = public_id_for_module(module_path)
    return MODEL_CATALOG.get(public_id) if public_id is not None else None


def public_id_for_flat_name(flat_name: str) -> str | None:
    return FLAT_TO_PUBLIC_ID.get(flat_name)


def identity_for_module(module_path: str) -> ModelIdentity | None:
    package = model_package_for_module(module_path)
    return package.identity if package is not None else None


def model_id_from_payload(payload: Mapping[str, Any]) -> str | None:
    model_type = payload.get("modelType")
    model = payload.get("model")
    if isinstance(model_type, str) and isinstance(model, str):
        return model_id_from_parts(model_type, model)
    if not isinstance(model, str):
        return None
    if model_package(model) is not None:
        return model
    return public_id_for_flat_name(model)


def model_identity_payload_from_id(model_id: str) -> dict[str, str]:
    package = model_package(model_id)
    if package is not None:
        return package.to_identity_payload()
    identity = split_model_id(model_id)
    if identity is None:
        raise ValueError(f"Invalid model id: {model_id}")
    return identity.to_payload()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve Emperor model catalog entries."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="Print public model flags.")
    action.add_argument(
        "--list-types",
        action="store_true",
        help="Print available model types.",
    )
    action.add_argument(
        "--module",
        action="store_true",
        help="Print a model module path.",
    )
    parser.add_argument("--model-type", help="Model type, e.g. linears.")
    parser.add_argument("--model", help="Model name, e.g. linear.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_types:
        for model_type in discover_model_types():
            print(model_type)
        return

    if args.list:
        if args.model_type:
            if not model_type_exists(args.model_type):
                raise SystemExit(f"Unknown model type: --model-type {args.model_type}")
            for identity in discover_model_identities_for_type(args.model_type):
                print(f"--model {identity.model}")
            return

        for identity in discover_model_identities():
            print(f"--model-type {identity.model_type} --model {identity.model}")
        return

    if not args.model_type or not args.model:
        raise SystemExit("--module requires --model-type and --model.")
    module_path = module_path_for_model_identity(args.model_type, args.model)
    if module_path is None:
        raise SystemExit(1)
    print(module_path)


__all__ = [
    "EMPTY_CATEGORY_PACKAGES",
    "FLAT_TO_PUBLIC_ID",
    "MODEL_CATALOG",
    "MODEL_ID_SEGMENT_RE",
    "MODEL_ORDER",
    "ModelCatalogEntry",
    "ModelIdentity",
    "ModelPackage",
    "catalog_entry",
    "discover_model_identities",
    "discover_model_identities_for_type",
    "discover_model_identity_payloads",
    "discover_model_ids",
    "discover_model_packages",
    "discover_model_types",
    "identity_for_module",
    "is_safe_model_id",
    "is_safe_model_identity",
    "is_safe_model_segment",
    "model_id_from_parts",
    "model_id_from_payload",
    "model_identity_for_model_id",
    "model_identity_for_parts",
    "model_identity_payload",
    "model_identity_payload_from_id",
    "model_key",
    "model_package",
    "model_package_for_module",
    "model_type_exists",
    "module_path_for_model_id",
    "module_path_for_model_identity",
    "public_id_for_flat_name",
    "public_id_for_module",
    "split_model_id",
]


if __name__ == "__main__":
    main()
