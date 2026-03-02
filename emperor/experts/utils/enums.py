from enum import Enum


class ExpertWeightingPositionOptions(Enum):
    BEFORE_EXPERTS = 1
    AFTER_EXPERTS = 2


class InitSamplerOptions(Enum):
    DISABLED = 1
    SHARED = 2
    LAYER = 3
