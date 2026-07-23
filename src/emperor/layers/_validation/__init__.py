from emperor.layers._validation.attention_residual import (
    AttentionResidualValidator,
)
from emperor.layers._validation.common import (
    _config_classes,
    _gate_config_class,
    _gate_option_field_path,
    _linear_layer_config_class,
    _matches_config_contract,
    _residual_config_class,
    _validate_halting_lifecycle_owner,
)
from emperor.layers._validation.gate import LayerGateValidator
from emperor.layers._validation.layer import LayerValidator
from emperor.layers._validation.recurrent import RecurrentLayerValidator
from emperor.layers._validation.residual import ResidualConnectionValidator
from emperor.layers._validation.stack import LayerStackValidator

__all__ = [
    "AttentionResidualValidator",
    "LayerGateValidator",
    "LayerStackValidator",
    "LayerValidator",
    "RecurrentLayerValidator",
    "ResidualConnectionValidator",
]

for _validator in (
    AttentionResidualValidator,
    LayerGateValidator,
    LayerStackValidator,
    LayerValidator,
    RecurrentLayerValidator,
    ResidualConnectionValidator,
):
    _validator.__module__ = __name__

del _validator

for _helper in (
    _config_classes,
    _gate_config_class,
    _gate_option_field_path,
    _linear_layer_config_class,
    _matches_config_contract,
    _residual_config_class,
    _validate_halting_lifecycle_owner,
):
    _helper.__module__ = __name__

del _helper
