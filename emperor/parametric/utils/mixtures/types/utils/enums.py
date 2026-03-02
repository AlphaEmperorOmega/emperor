from emperor.base.enums import BaseOptions


class ClipParameterOptions(BaseOptions):
    NONE = "no_clipping"
    BEFORE = "clip_parameter_vectors"
    AFTER = "clip_by_top_k_weight"
