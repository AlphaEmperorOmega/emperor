import torch

from torch import Tensor
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase
from Emperor.datasets.image.mnist import Mnist
from models.parser import get_experiment_parser
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.base.layer import LayerStack, LayerStackConfig
from Emperor.experiments.classifier import ClassifierExperiment
from Emperor.transformer.utils.layers import TransformerConfig
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.transformer.utils.patch.selector import PatchOptions
from Emperor.transformer.utils.patch.selector import PatchSelector
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.transformer.utils.stack import TransformerEncoderStack
from Emperor.transformer.utils.patch.options.base import PatchConfig
from Emperor.transformer.utils.feed_forward import FeedForwardConfig
from Emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from Emperor.embedding.options import AbsolutePositionalEmbeddingOptions
from Emperor.embedding.absolute.factory import AbsolutePositionalEmbeddingFactory
from Emperor.embedding.absolute.config import AbsolutePositionalEmbeddingConfig
from Emperor.base.enums import ActivationOptions, BaseOptions, LayerNormPositionOptions
from Emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
)
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class ExperimentConfig(ConfigBase):
    patch_config: "PatchConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    positional_embedding_config: "AbsolutePositionalEmbeddingConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    encoder_config: "TransformerConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)

        self.patch_config = self.main_cfg.patch_config
        self.embedding_config = self.main_cfg.positional_embedding_config
        self.encoder_config = self.main_cfg.encoder_config
        self.output_config = self.main_cfg.output_config

        self.patch = PatchSelector(self.patch_config).build()
        self.positional_embedding = AbsolutePositionalEmbeddingFactory(
            self.embedding_config
        ).build()
        self.transformer = TransformerEncoderStack(self.encoder_config)
        self.output = LayerStack(self.output_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> "ExperimentConfig":
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = self.patch(X)
        X = self.positional_embedding(X)
        X, loss = self.transformer(X)
        X = X[:, 0, :]
        X = self.output(X)
        return X


class ExperimentOptions(BaseOptions):
    DEFAULT = 0
    BASE = 1
    ADAPTIVE = 2


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.DEFAULT,
        dataset: type = Mnist,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.DEFAULT:
                return self._default_config(dataset)
            case ExperimentOptions.BASE:
                return [self._preset(**self._dataset_config(dataset))]
            case ExperimentOptions.ADAPTIVE:
                return [self.__adaptive_preset(**self._dataset_config(dataset))]
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def _preset(
        self,
        batch_size: int = 64,
        input_dim: int = 32,
        hidden_dim: int = 32,
        output_dim: int = 10,
        dropout_probability: float = 0.1,
        activation_function: ActivationOptions = ActivationOptions.SILU,
        output_num_layers: int = 2,
        transformer_num_layers: int = 1,
        attn_bias_flag: bool = False,
        attn_num_heads: int = 4,
        attn_model_type: LinearLayerStackOptions = LinearLayerStackOptions.BASE,
        attn_num_layers: int = 1,
        ff_bias_flag: bool = True,
        ff_model_type: LinearLayerStackOptions = LinearLayerStackOptions.BASE,
        ff_num_layers: int = 2,
        output_bias_flag: bool = True,
        image_patch_size: int = 4,
        input_channels: int = 3,
        image_height: int = 32,
    ) -> "ModelConfig":
        from Emperor.config import ModelConfig

        class_token_length = 1
        padding_size = 0
        dilatation_size = 0
        stride = image_patch_size
        h_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        w_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        sequence_length = h_out * w_out + 1
        num_embeddings = sequence_length - class_token_length

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.CONV,
                    embedding_dim=hidden_dim,
                    num_input_channels=input_channels,
                    patch_size=image_patch_size,
                    stride=image_patch_size,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_layers=1,
                        activation=activation_function,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=dropout_probability,
                        override_config=LinearLayerConfig(
                            input_dim=hidden_dim,
                            output_dim=hidden_dim,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=AbsolutePositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=AbsolutePositionalEmbeddingOptions.LEARNED,
                    num_embeddings=num_embeddings,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerConfig(
                    num_layers=transformer_num_layers,
                    source_sequence_length=sequence_length,
                    target_sequence_length=sequence_length,
                    embedding_dim=hidden_dim,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    dropout_probability=dropout_probability,
                    causal_attention_mask_flag=False,
                    attention_config=MultiHeadAttentionConfig(
                        batch_size=batch_size,
                        num_heads=attn_num_heads,
                        model_type=attn_model_type,
                        query_key_projection_dim=hidden_dim,
                        value_projection_dim=hidden_dim,
                        embedding_dim=hidden_dim,
                        target_sequence_length=sequence_length,
                        source_sequence_length=sequence_length,
                        target_dtype=torch.float32,
                        attention_option=True,
                        dropout_probability=dropout_probability,
                        key_value_bias_flag=False,
                        zero_attention_flag=False,
                        causal_attention_mask_flag=False,
                        add_key_value_bias_flag=False,
                        average_attention_weights_flag=False,
                        return_attention_weights_flag=False,
                        override_config=LayerStackConfig(
                            input_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            output_dim=hidden_dim,
                            num_layers=attn_num_layers,
                            activation=activation_function,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=hidden_dim,
                                output_dim=hidden_dim,
                                bias_flag=attn_bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                    feed_forward_config=FeedForwardConfig(
                        layer_stack_option=ff_model_type,
                        input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_layers=ff_num_layers,
                        override_config=LayerStackConfig(
                            input_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            output_dim=hidden_dim,
                            num_layers=ff_num_layers,
                            activation=activation_function,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=hidden_dim,
                                output_dim=hidden_dim,
                                bias_flag=ff_bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                ),
                output_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=output_num_layers,
                    activation=activation_function,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=False,
                    adaptive_computation_flag=False,
                    dropout_probability=dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=hidden_dim,
                        output_dim=output_dim,
                        bias_flag=output_bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            ),
        )

    def __adaptive_preset(
        self,
        batch_size: int = 64,
        input_dim: int = 32,
        hidden_dim: int = 32,
        output_dim: int = 10,
        dropout_probability: float = 0.1,
        activation_function: ActivationOptions = ActivationOptions.SILU,
        transformer_num_layers: int = 1,
        attn_bias_flag: bool = True,
        attn_num_layers: int = 1,
        attn_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        attn_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL,
        attn_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        attn_behaviour_stack_num_layers: int = 2,
        ff_bias_flag: bool = True,
        ff_num_layers: int = 2,
        ff_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        ff_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL,
        ff_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        ff_behaviour_stack_num_layers: int = 2,
        output_bias_flag: bool = True,
        output_num_layers: int = 1,
        output_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        output_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL,
        output_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        output_behaviour_stack_num_layers: int = 2,
        image_patch_size: int = 4,
        input_channels: int = 3,
        image_height: int = 32,
    ) -> "ModelConfig":
        from Emperor.config import ModelConfig
        from Emperor.linears.utils.presets import LinearPresets

        class_token_length = 1
        padding_size = 0
        dilatation_size = 0
        stride = image_patch_size
        h_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        w_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        sequence_length = h_out * w_out + 1
        num_embeddings = sequence_length - class_token_length

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.CONV,
                    num_input_channels=input_channels,
                    embedding_dim=hidden_dim,
                    patch_size=image_patch_size,
                    stride=image_patch_size,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_layers=1,
                        activation=activation_function,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=dropout_probability,
                        override_config=LinearLayerConfig(
                            input_dim=hidden_dim,
                            output_dim=hidden_dim,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=AbsolutePositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=AbsolutePositionalEmbeddingOptions.LEARNED,
                    num_embeddings=num_embeddings,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerPresets.transformer_linear_adaptive_preset(
                    batch_size=batch_size,
                    num_layers=transformer_num_layers,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    num_heads=4,
                    embedding_dim=hidden_dim,
                    query_key_projection_dim=hidden_dim,
                    value_projection_dim=hidden_dim,
                    target_sequence_length=sequence_length,
                    source_sequence_length=sequence_length,
                    target_dtype=torch.float32,
                    attention_option=False,
                    dropout_probability=dropout_probability,
                    key_value_bias_flag=False,
                    zero_attention_flag=False,
                    causal_attention_mask_flag=False,
                    add_key_value_bias_flag=False,
                    average_attention_weights_flag=False,
                    return_attention_weights_flag=False,
                    attn_stack_num_layers=attn_num_layers,
                    attn_bias_flag=attn_bias_flag,
                    attn_generator_depth=attn_generator_depth,
                    attn_diagonal_option=attn_diagonal_option,
                    attn_bias_option=attn_bias_option,
                    attn_behaviour_stack_num_layers=attn_behaviour_stack_num_layers,
                    ff_stack_num_layers=ff_num_layers,
                    ff_bias_flag=ff_bias_flag,
                    ff_generator_depth=ff_generator_depth,
                    ff_diagonal_option=ff_diagonal_option,
                    ff_bias_option=ff_bias_option,
                    ff_behaviour_stack_num_layers=ff_behaviour_stack_num_layers,
                    stack_activation=activation_function,
                    stack_residual_flag=False,
                ),
                output_config=LinearPresets.adaptive_linear_layer_stack_preset(
                    batch_size=batch_size,
                    input_dim=hidden_dim,
                    output_dim=output_dim,
                    bias_flag=output_bias_flag,
                    generator_depth=output_generator_depth,
                    diagonal_option=output_diagonal_option,
                    bias_option=output_bias_option,
                    stack_num_layers=output_num_layers,
                    stack_hidden_dim=hidden_dim,
                    stack_activation=activation_function,
                    stack_residual_flag=False,
                    stack_dropout_probability=dropout_probability,
                    adaptive_behaviour_stack_num_layers=output_behaviour_stack_num_layers,
                ),
            ),
        )


if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option = ExperimentOptions.get_option(args.name)

    experiment = Experiment(config_option)
    experiment.train_model()
