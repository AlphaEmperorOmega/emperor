from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

MODEL_ID_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModelCatalogEntry:
    public_id: str
    module_path: str


MODEL_CATALOG: dict[str, ModelCatalogEntry] = {
    "linears/linear": ModelCatalogEntry(
        public_id="linears/linear",
        module_path="models.linears.linear",
    ),
    "linears/linear_adaptive": ModelCatalogEntry(
        public_id="linears/linear_adaptive",
        module_path="models.linears.linear_adaptive",
    ),
    "experts/experts_linear": ModelCatalogEntry(
        public_id="experts/experts_linear",
        module_path="models.experts.experts_linear",
    ),
    "experts/experts_linear_adaptive": ModelCatalogEntry(
        public_id="experts/experts_linear_adaptive",
        module_path="models.experts.experts_linear_adaptive",
    ),
    "parametric/parametric_vector": ModelCatalogEntry(
        public_id="parametric/parametric_vector",
        module_path="models.parametric.parametric_vector",
    ),
    "parametric/parametric_matrix": ModelCatalogEntry(
        public_id="parametric/parametric_matrix",
        module_path="models.parametric.parametric_matrix",
    ),
    "parametric/parametric_generator": ModelCatalogEntry(
        public_id="parametric/parametric_generator",
        module_path="models.parametric.parametric_generator",
    ),
    "neuron/neuron_linear": ModelCatalogEntry(
        public_id="neuron/neuron_linear",
        module_path="models.neuron.neuron_linear",
    ),
    "transformer_encoder/bert_linear": ModelCatalogEntry(
        public_id="transformer_encoder/bert_linear",
        module_path="models.transformer_encoder.bert_linear",
    ),
    "transformer_encoder/vit_linear": ModelCatalogEntry(
        public_id="transformer_encoder/vit_linear",
        module_path="models.transformer_encoder.vit_linear",
    ),
}

EMPTY_CATEGORY_PACKAGES = {
    "models.transformer",
    "models.transformer_decoder",
}

_MODULE_TO_PUBLIC_ID = {
    entry.module_path: public_id for public_id, entry in MODEL_CATALOG.items()
}

FLAT_TO_PUBLIC_ID = {
    public_id.rsplit("/", 1)[-1]: public_id for public_id in MODEL_CATALOG
}


def discover_model_ids() -> list[str]:
    return sorted(MODEL_CATALOG)


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


def public_id_for_module(module_path: str) -> str | None:
    if module_path in _MODULE_TO_PUBLIC_ID:
        return _MODULE_TO_PUBLIC_ID[module_path]
    for package, public_id in _MODULE_TO_PUBLIC_ID.items():
        if module_path.startswith(f"{package}."):
            return public_id
    return None


def public_id_for_flat_name(flat_name: str) -> str | None:
    return FLAT_TO_PUBLIC_ID.get(flat_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve Emperor model catalog entries."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="Print public model IDs.")
    action.add_argument(
        "--module",
        metavar="MODEL_ID",
        help="Print a model module path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list:
        for model_id in discover_model_ids():
            print(model_id)
        return

    module_path = module_path_for_model_id(args.module)
    if module_path is None:
        raise SystemExit(1)
    print(module_path)


if __name__ == "__main__":
    main()
