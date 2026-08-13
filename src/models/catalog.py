from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
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
from models.mlp_mixer.expert_linear import MODEL_PACKAGE as MLP_MIXER_EXPERT_LINEAR
from models.mlp_mixer.expert_linear_adaptive import (
    MODEL_PACKAGE as MLP_MIXER_EXPERT_LINEAR_ADAPTIVE,
)
from models.mlp_mixer.linear import MODEL_PACKAGE as MLP_MIXER_LINEAR
from models.mlp_mixer.linear_adaptive import (
    MODEL_PACKAGE as MLP_MIXER_LINEAR_ADAPTIVE,
)
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


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """One validated Model Package registration and its family display order."""

    package: ModelPackage
    display_order: int

    def __post_init__(self) -> None:
        if type(self.display_order) is not int:
            raise TypeError("Model registration display order must be an integer")
        if self.display_order < 0:
            raise ValueError("Model registration display order must be non-negative")


class ModelCatalog(Mapping[str, ModelPackage]):
    """Immutable, validated registry of selectable Model Packages."""

    def __init__(self, registrations: Iterable[ModelRegistration]) -> None:
        resolved_registrations = tuple(registrations)
        packages: dict[str, ModelPackage] = {}
        orders: dict[str, int] = {}
        ordered_identity_by_family: dict[tuple[str, int], str] = {}

        for registration in resolved_registrations:
            package = registration.package
            catalog_key = package.identity.catalog_key
            if catalog_key in packages:
                raise ValueError(
                    f"duplicate Model Package identity in catalog: {catalog_key}"
                )
            family_order = (package.identity.model_type, registration.display_order)
            existing_identity = ordered_identity_by_family.get(family_order)
            if existing_identity is not None:
                raise ValueError(
                    "duplicate Model Package display order "
                    f"{registration.display_order} for {package.identity.model_type}: "
                    f"{existing_identity}, {catalog_key}"
                )
            packages[catalog_key] = package
            orders[catalog_key] = registration.display_order
            ordered_identity_by_family[family_order] = catalog_key

        self.__registrations = resolved_registrations
        self.__packages = MappingProxyType(packages)
        self.__orders = MappingProxyType(orders)

    @property
    def registrations(self) -> tuple[ModelRegistration, ...]:
        return self.__registrations

    @property
    def display_orders(self) -> Mapping[str, int]:
        return self.__orders

    def __getitem__(self, catalog_key: str) -> ModelPackage:
        return self.__packages[catalog_key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__packages)

    def __len__(self) -> int:
        return len(self.__packages)

    def discover_model_ids(self) -> list[str]:
        return sorted(
            self.__packages,
            key=lambda key: (
                self.__packages[key].identity.model_type,
                self.__orders[key],
            ),
        )

    def discover_model_packages(self) -> list[ModelPackage]:
        return [self.__packages[key] for key in self.discover_model_ids()]

    def discover_model_types(self) -> list[str]:
        return sorted(
            {package.identity.model_type for package in self.__packages.values()}
        )


_MODEL_REGISTRATIONS = (
    ModelRegistration(BERT_LINEAR, 0),
    ModelRegistration(BERT_LINEAR_ADAPTIVE, 1),
    ModelRegistration(BERT_EXPERT_LINEAR, 2),
    ModelRegistration(BERT_EXPERT_LINEAR_ADAPTIVE, 3),
    ModelRegistration(GPT_LINEAR, 0),
    ModelRegistration(GPT_LINEAR_ADAPTIVE, 1),
    ModelRegistration(GPT_EXPERT_LINEAR, 2),
    ModelRegistration(GPT_EXPERT_LINEAR_ADAPTIVE, 3),
    ModelRegistration(VIT_LINEAR, 0),
    ModelRegistration(VIT_LINEAR_ADAPTIVE, 1),
    ModelRegistration(VIT_EXPERT_LINEAR, 2),
    ModelRegistration(VIT_EXPERT_LINEAR_ADAPTIVE, 3),
    ModelRegistration(MLP_MIXER_LINEAR, 0),
    ModelRegistration(MLP_MIXER_LINEAR_ADAPTIVE, 1),
    ModelRegistration(MLP_MIXER_EXPERT_LINEAR, 2),
    ModelRegistration(MLP_MIXER_EXPERT_LINEAR_ADAPTIVE, 3),
    ModelRegistration(TRANSFORMER_LINEAR, 0),
    ModelRegistration(TRANSFORMER_LINEAR_ADAPTIVE, 1),
    ModelRegistration(TRANSFORMER_EXPERT_LINEAR, 2),
    ModelRegistration(TRANSFORMER_EXPERT_LINEAR_ADAPTIVE, 3),
    ModelRegistration(LINEARS_LINEAR, 0),
    ModelRegistration(LINEARS_LINEAR_ADAPTIVE, 1),
    ModelRegistration(EXPERTS_LINEAR, 0),
    ModelRegistration(EXPERTS_LINEAR_ADAPTIVE, 1),
    ModelRegistration(PARAMETRIC_VECTOR, 2),
    ModelRegistration(PARAMETRIC_MATRIX, 1),
    ModelRegistration(PARAMETRIC_GENERATOR, 0),
    ModelRegistration(NEURON_LINEAR, 0),
    ModelRegistration(NEURON_LINEAR_ADAPTIVE, 1),
    ModelRegistration(NEURON_EXPERT_LINEAR, 2),
    ModelRegistration(NEURON_EXPERT_LINEAR_ADAPTIVE, 3),
)

MODEL_CATALOG = ModelCatalog(_MODEL_REGISTRATIONS)
MODEL_ORDER = MODEL_CATALOG.display_orders

EMPTY_CATEGORY_PACKAGES: frozenset[str] = frozenset()


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
    return MODEL_CATALOG.discover_model_ids()


def discover_model_packages() -> list[ModelPackage]:
    return MODEL_CATALOG.discover_model_packages()


def discover_model_types() -> list[str]:
    return MODEL_CATALOG.discover_model_types()


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
    "ModelCatalog",
    "ModelIdentity",
    "ModelPackage",
    "ModelRegistration",
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
