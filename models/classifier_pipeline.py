from emperor.base.utils import ConfigBase, Module


def build_from_experiment_config(
    model_config: ConfigBase,
    *,
    input_dim: int,
    output_dim: int,
) -> Module:
    override = type(model_config)(
        input_dim=input_dim,
        output_dim=output_dim,
    )
    return model_config.build(overrides=override)
