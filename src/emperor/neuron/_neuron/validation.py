from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.neuron._axons.validation import AxonsValidator
from emperor.neuron._terminal.validation import Validator
from emperor.neuron._validation.common import NeuronValidationMixin

if TYPE_CHECKING:
    from emperor.neuron._config import NeuronConfig
    from emperor.neuron._neuron.core import Neuron


class NeuronValidator(ValidatorBase, NeuronValidationMixin):
    AXONS_VALIDATOR = AxonsValidator
    TERMINAL_VALIDATOR = Validator

    OPTIONAL_FIELDS = {"coordinate_embedding_flag"}

    @classmethod
    def validate(cls, cfg: "NeuronConfig") -> None:
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_coordinate_embedding_options(cfg)
        cls.validate_nucleus_model_dimensions(cfg)
        cls.validate_axons_memory_dimensions(cfg)
        cls.validate_terminal_composition(cfg)

    @staticmethod
    def validate_coordinate_embedding_options(cfg: "NeuronConfig") -> None:
        coordinate_embedding_flag = cfg.coordinate_embedding_flag
        if coordinate_embedding_flag is None:
            return
        if not isinstance(coordinate_embedding_flag, bool):
            raise TypeError(
                "coordinate_embedding_flag must be a bool for NeuronConfig, "
                f"got {type(coordinate_embedding_flag).__name__}."
            )
        if not coordinate_embedding_flag:
            return
        terminal_input_dim = cfg.terminal_config.input_dim
        if terminal_input_dim < 3:
            raise ValueError(
                "coordinate_embedding_flag requires terminal_config.input_dim "
                "of at least 3 so every coordinate axis receives at least one "
                f"encoding channel, received input_dim={terminal_input_dim}."
            )

    @staticmethod
    def validate_nucleus_model_dimensions(cfg: "NeuronConfig") -> None:
        nucleus_model_config = cfg.nucleus_config.model_config
        terminal_input_dim = cfg.terminal_config.input_dim
        for dimension_name in ("input_dim", "output_dim"):
            configured_dimension = getattr(
                nucleus_model_config,
                dimension_name,
                None,
            )
            if configured_dimension is None:
                continue
            if configured_dimension != terminal_input_dim:
                raise ValueError(
                    "nucleus_config.model_config must preserve the terminal "
                    "feature dimension for NeuronConfig, received "
                    f"{dimension_name}={configured_dimension} and terminal "
                    f"input_dim={terminal_input_dim}."
                )

    @classmethod
    def validate_axons_memory_dimensions(cls, cfg: "NeuronConfig") -> None:
        cls.AXONS_VALIDATOR.validate_config(cfg.axons_config)
        memory_config = cfg.axons_config.memory_config
        if memory_config is None:
            return
        if memory_config.input_dim is None:
            raise ValueError(
                "axons_config.memory_config.input_dim is required for NeuronConfig, "
                "received None."
            )
        terminal_input_dim = cfg.terminal_config.input_dim
        if memory_config.input_dim != terminal_input_dim:
            raise ValueError(
                "axons_config.memory_config.input_dim must preserve the terminal "
                "feature dimension for NeuronConfig, received memory input_dim="
                f"{memory_config.input_dim} and terminal input_dim="
                f"{terminal_input_dim}."
            )

    @classmethod
    def validate_terminal_composition(cls, cfg: "NeuronConfig") -> None:
        cls.TERMINAL_VALIDATOR.validate_config_composition(cfg.terminal_config)

    @classmethod
    def validate_forward_input(cls, input: Tensor) -> None:
        cls.validate_tensor_rank("Neuron input", input, 2)

    @staticmethod
    def validate_feature_dimension(model: "Neuron", input: Tensor) -> None:
        input_dim = model.cfg.terminal_config.input_dim
        if input.shape[-1] != input_dim:
            raise ValueError(
                "Neuron input feature dimension must match terminal_config.input_dim, "
                f"received input_dim={input_dim} and input shape {tuple(input.shape)}."
            )
