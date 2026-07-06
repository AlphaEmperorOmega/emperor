from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MODEL_ID_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModelCatalogEntry:
    model_type: str
    model: str
    module_path: str

    @property
    def catalog_key(self) -> str:
        return model_key(self.model_type, self.model)

    @property
    def public_id(self) -> str:
        """Legacy slash-joined id retained as a private catalog/import key."""

        return self.catalog_key

    def to_identity_payload(self) -> dict[str, str]:
        return {
            "modelType": self.model_type,
            "model": self.model,
        }


@dataclass(frozen=True)
class ModelIdentity:
    model_type: str
    model: str

    @property
    def catalog_key(self) -> str:
        return model_key(self.model_type, self.model)

    def to_payload(self) -> dict[str, str]:
        return {
            "modelType": self.model_type,
            "model": self.model,
        }


MODEL_CATALOG: dict[str, ModelCatalogEntry] = {
    "bert/linear": ModelCatalogEntry(
        model_type="bert",
        model="linear",
        module_path="models.bert.linear",
    ),
    "bert/linear_adaptive": ModelCatalogEntry(
        model_type="bert",
        model="linear_adaptive",
        module_path="models.bert.linear_adaptive",
    ),
    "bert/expert_linear": ModelCatalogEntry(
        model_type="bert",
        model="expert_linear",
        module_path="models.bert.expert_linear",
    ),
    "bert/expert_linear_adaptive": ModelCatalogEntry(
        model_type="bert",
        model="expert_linear_adaptive",
        module_path="models.bert.expert_linear_adaptive",
    ),
    "vit/linear": ModelCatalogEntry(
        model_type="vit",
        model="linear",
        module_path="models.vit.linear",
    ),
    "vit/linear_adaptive": ModelCatalogEntry(
        model_type="vit",
        model="linear_adaptive",
        module_path="models.vit.linear_adaptive",
    ),
    "vit/expert_linear": ModelCatalogEntry(
        model_type="vit",
        model="expert_linear",
        module_path="models.vit.expert_linear",
    ),
    "vit/expert_linear_adaptive": ModelCatalogEntry(
        model_type="vit",
        model="expert_linear_adaptive",
        module_path="models.vit.expert_linear_adaptive",
    ),
    "linears/linear": ModelCatalogEntry(
        model_type="linears",
        model="linear",
        module_path="models.linears.linear",
    ),
    "linears/linear_adaptive": ModelCatalogEntry(
        model_type="linears",
        model="linear_adaptive",
        module_path="models.linears.linear_adaptive",
    ),
    "experts/linear": ModelCatalogEntry(
        model_type="experts",
        model="linear",
        module_path="models.experts.linear",
    ),
    "experts/linear_adaptive": ModelCatalogEntry(
        model_type="experts",
        model="linear_adaptive",
        module_path="models.experts.linear_adaptive",
    ),
    "parametric/parametric_vector": ModelCatalogEntry(
        model_type="parametric",
        model="parametric_vector",
        module_path="models.parametric.parametric_vector",
    ),
    "parametric/parametric_matrix": ModelCatalogEntry(
        model_type="parametric",
        model="parametric_matrix",
        module_path="models.parametric.parametric_matrix",
    ),
    "parametric/parametric_generator": ModelCatalogEntry(
        model_type="parametric",
        model="parametric_generator",
        module_path="models.parametric.parametric_generator",
    ),
    "neuron/linear": ModelCatalogEntry(
        model_type="neuron",
        model="linear",
        module_path="models.neuron.linear",
    ),
    "neuron/linear_adaptive": ModelCatalogEntry(
        model_type="neuron",
        model="linear_adaptive",
        module_path="models.neuron.linear_adaptive",
    ),
    "neuron/expert_linear": ModelCatalogEntry(
        model_type="neuron",
        model="expert_linear",
        module_path="models.neuron.expert_linear",
    ),
    "neuron/expert_linear_adaptive": ModelCatalogEntry(
        model_type="neuron",
        model="expert_linear_adaptive",
        module_path="models.neuron.expert_linear_adaptive",
    ),
}

MODEL_ORDER: dict[str, int] = {
    "bert/linear": 0,
    "bert/linear_adaptive": 1,
    "bert/expert_linear": 2,
    "bert/expert_linear_adaptive": 3,
    "vit/linear": 0,
    "vit/linear_adaptive": 1,
    "vit/expert_linear": 2,
    "vit/expert_linear_adaptive": 3,
    "neuron/linear": 0,
    "neuron/linear_adaptive": 1,
    "neuron/expert_linear": 2,
    "neuron/expert_linear_adaptive": 3,
}

EMPTY_CATEGORY_PACKAGES = {
    "models.transformer",
}

_MODULE_TO_PUBLIC_ID = {
    entry.module_path: public_id for public_id, entry in MODEL_CATALOG.items()
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
    # Preserve legacy flat log names for the original catalog owner when names collide.
    FLAT_TO_PUBLIC_ID.setdefault(flat_name, public_id)


def is_safe_model_segment(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not value or value.strip() != value:
        return False
    return bool(MODEL_ID_SEGMENT_RE.fullmatch(value))


def is_safe_model_identity(model_type: str, model: str) -> bool:
    return is_safe_model_segment(model_type) and is_safe_model_segment(model)


def model_key(model_type: str, model: str) -> str:
    if not is_safe_model_identity(model_type, model):
        raise ValueError(f"Invalid model identity: {model_type!r}/{model!r}")
    return f"{model_type}/{model}"


def split_model_id(model_id: str) -> ModelIdentity | None:
    if not is_safe_model_id(model_id):
        return None
    segments = model_id.split("/")
    if len(segments) != 2:
        return None
    return ModelIdentity(model_type=segments[0], model=segments[1])


def model_identity_payload(model_id: str) -> dict[str, str]:
    identity = model_identity_for_model_id(model_id)
    if identity is None:
        legacy_identity = split_model_id(model_id)
        if legacy_identity is None:
            raise ValueError(f"Invalid model id: {model_id}")
        identity = legacy_identity
    return identity.to_payload()


def model_identity_for_model_id(model_id: str) -> ModelIdentity | None:
    entry = catalog_entry(model_id)
    if entry is None:
        return None
    return ModelIdentity(entry.model_type, entry.model)


def model_identity_for_parts(model_type: str, model: str) -> ModelIdentity | None:
    if not is_safe_model_identity(model_type, model):
        return None
    key = model_key(model_type, model)
    entry = MODEL_CATALOG.get(key)
    if entry is None:
        return None
    return ModelIdentity(entry.model_type, entry.model)


def model_id_from_parts(model_type: str, model: str) -> str | None:
    identity = model_identity_for_parts(model_type, model)
    return identity.catalog_key if identity is not None else None


def discover_model_ids() -> list[str]:
    return sorted(
        MODEL_CATALOG,
        key=lambda key: (key.split("/", 1)[0], MODEL_ORDER.get(key, key)),
    )


def discover_model_types() -> list[str]:
    return sorted({entry.model_type for entry in MODEL_CATALOG.values()})


def model_type_exists(model_type: str) -> bool:
    if not is_safe_model_segment(model_type):
        return False
    return model_type in discover_model_types()


def discover_model_identities() -> list[ModelIdentity]:
    return [
        ModelIdentity(MODEL_CATALOG[key].model_type, MODEL_CATALOG[key].model)
        for key in discover_model_ids()
    ]


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


def is_safe_model_id(model_id: str) -> bool:
    if not isinstance(model_id, str):
        return False
    if not model_id or model_id.strip() != model_id:
        return False
    if "\\" in model_id:
        return False
    segments = model_id.split("/")
    if len(segments) < 2:
        return False
    return all(
        segment not in {".", ".."} and MODEL_ID_SEGMENT_RE.fullmatch(segment)
        for segment in segments
    )


def catalog_entry(model_id: str) -> ModelCatalogEntry | None:
    if not is_safe_model_id(model_id):
        return None
    return MODEL_CATALOG.get(model_id)


def module_path_for_model_id(model_id: str) -> str | None:
    entry = catalog_entry(model_id)
    return entry.module_path if entry is not None else None


def module_path_for_model_identity(model_type: str, model: str) -> str | None:
    model_id = model_id_from_parts(model_type, model)
    if model_id is None:
        return None
    return module_path_for_model_id(model_id)


def public_id_for_module(module_path: str) -> str | None:
    if module_path in _MODULE_TO_PUBLIC_ID:
        return _MODULE_TO_PUBLIC_ID[module_path]
    for package, public_id in _MODULE_TO_PUBLIC_ID.items():
        if module_path.startswith(f"{package}."):
            return public_id
    return None


def public_id_for_flat_name(flat_name: str) -> str | None:
    return FLAT_TO_PUBLIC_ID.get(flat_name)


def identity_for_module(module_path: str) -> ModelIdentity | None:
    public_id = public_id_for_module(module_path)
    if public_id is None:
        return None
    return model_identity_for_model_id(public_id)


def model_id_from_payload(payload: Mapping[str, Any]) -> str | None:
    model_type = payload.get("modelType")
    model = payload.get("model")
    if isinstance(model_type, str) and isinstance(model, str):
        return model_id_from_parts(model_type, model)
    if not isinstance(model, str):
        return None
    if catalog_entry(model) is not None:
        return model
    return public_id_for_flat_name(model)


def model_identity_payload_from_id(model_id: str) -> dict[str, str]:
    entry = catalog_entry(model_id)
    if entry is not None:
        return entry.to_identity_payload()
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


if __name__ == "__main__":
    main()
