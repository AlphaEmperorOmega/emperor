from enum import Enum


class AttentionOptions(Enum):
    SELF_ATTENTION = 1
    INDEPENDENT = 2
    MIXTURE_OF_ATTENTION_HEADS = 3
