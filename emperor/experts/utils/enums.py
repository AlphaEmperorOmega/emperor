from enum import Enum


class ExpertWeightingPositionOptions(Enum):
    BEFORE_EXPERTS = 1
    AFTER_EXPERTS = 2


class InitSamplerOptions(Enum):
    DISABLED = 1
    SHARED = 2
    LAYER = 3


class DroppedTokenOptions(Enum):
    ZERO = 1
    IDENTITY = 2
