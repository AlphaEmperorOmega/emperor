from enum import Enum


class ExpertWeightingPositionOptions(Enum):
    BEFORE_EXPERTS = 1
    AFTER_EXPERTS = 2


class LayerRoleOptions(Enum):
    GENERAL = 1
    INPUT = 2
    OUTPUT = 3
