"""Public Runtime Defaults Interface for the BERT expert-adaptive Model Package."""

from typing import Final

from models.bert.expert_linear_adaptive._option_resolution import (
    expert_linear_adaptive_builder_kwargs_from_flat as expert_linear_adaptive_builder_kwargs_from_flat,
)
from models.bert.expert_linear_adaptive._option_resolution import (
    expert_linear_builder_kwargs_from_flat as expert_linear_builder_kwargs_from_flat,
)
from models.bert.expert_linear_adaptive._option_resolution import (
    runtime_from_flat,
)
from models.bert.expert_linear_adaptive.runtime_options import RuntimeOptions

DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_flat"]
