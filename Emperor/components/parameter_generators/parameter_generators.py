class VectorParameter(ParameterBase):
    def __init__(
        self,
        cfg: "ParameterGeneratorConfig | ModelConfig",
        overrides: "ParameterGeneratorConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router: VectorRouterModel = VectorRouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: VectorMixture = VectorMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> VectorRouterModel | None:
        if self.bias_parameters_flag:
            return VectorRouterModel(cfg, self.bias_parameters_flag)
        return None

    def _compute_probabilities_and_indices(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self._compute_logits(input_batch, compute_bias_flag)
        input_dim, batch_size, depth_dim = logits.shape
        logits = logits.view(-1, depth_dim)

        probabilities, indices, _ = self.sampler.sample_probabilities_and_indices(
            logits
        )
        probabilities = probabilities.reshape(input_dim, batch_size, -1)
        indices = indices.reshape(input_dim, batch_size, -1)

        return probabilities, indices


