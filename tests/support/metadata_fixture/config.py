# ruff: noqa: E402, F401, F403, F811, I001

UNSECTIONED = 1

from support.metadata_fixture.base import *
from support.metadata_fixture.secondary import *
from support.metadata_fixture.recursive_a import *
from support.metadata_fixture.search_space import *
from support.metadata_fixture.explicit import (
    EXPLICIT_ONLY,
    IMPORTED_COLLISION,
    SHARED,
)

import support.metadata_fixture.aliased.config as aliased_config

_CONFIG_FIELD_METADATA_ALIASES = {
    "ALIAS_CHAIN": "SHARED",
    "ALIAS_TARGET": "ALIAS_CHAIN",
    "CYCLE_A": "CYCLE_B",
    "CYCLE_B": "CYCLE_A",
    "INVALID_ALIAS": 1,
}

# Local Options
SHARED = "local"
LOCAL_ONLY = 1
SEARCH_SPACE_LOCAL = (1, 2)
