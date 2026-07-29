from dataclasses import dataclass, replace


@dataclass(frozen=True)
class _ResolvedDatasetMetadata:
    """Schema dimensions resolved by one dataset adapter instance."""

    vocab_size: int | None = None
    num_classes: int | None = None
    flattened_input_dim: int | None = None

    def resolve(
        self,
        *,
        vocab_size: int | None = None,
        num_classes: int | None = None,
        flattened_input_dim: int | None = None,
    ) -> "_ResolvedDatasetMetadata":
        dimensions = {
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "flattened_input_dim": flattened_input_dim,
        }
        updates = {
            name: self._validated_dimension(name, value)
            for name, value in dimensions.items()
            if value is not None
        }
        return replace(self, **updates)

    @staticmethod
    def _validated_dimension(name: str, value: int) -> int:
        minimum = 0 if name == "num_classes" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(
                f"Resolved dataset {name} must be a {qualifier} integer."
            )
        return value
