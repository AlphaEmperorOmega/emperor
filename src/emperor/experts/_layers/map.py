from dataclasses import replace

from emperor.experts._config import MixtureOfExpertsConfig
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._options import RoutingInitializationMode


class MixtureOfExpertsMap(MixtureOfExperts):
    def __init__(
        self,
        cfg: MixtureOfExpertsConfig,
        overrides: MixtureOfExpertsConfig | None = None,
    ):
        self.VALIDATOR.validate_config_type(cfg)
        self.VALIDATOR.validate_overrides_type(overrides)
        overrides = self.__update_overrides(overrides)
        super().__init__(cfg, overrides)
        self.routing_positions = None
        self.sample_probabilities = None

    def __update_overrides(
        self, overrides: MixtureOfExpertsConfig | None = None
    ) -> MixtureOfExpertsConfig:
        if overrides is None:
            return MixtureOfExpertsConfig(
                weighted_parameters_flag=False,
                compute_expert_mixture_flag=False,
                routing_initialization_mode=RoutingInitializationMode.DISABLED,
            )
        return replace(
            overrides,
            weighted_parameters_flag=False,
            compute_expert_mixture_flag=False,
            routing_initialization_mode=RoutingInitializationMode.DISABLED,
        )
