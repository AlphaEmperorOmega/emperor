import torch

from torch import Tensor
from Emperor.config import ModelConfig
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.linears.utils.presets import LinearPresets
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.base.layer import LayerStack, LayerStackConfig
from Emperor.experiments.utils.factories import Experiments
from Emperor.transformer.utils.layers import TransformerConfig
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.transformer.utils.patch.selector import PatchOptions
from Emperor.transformer.utils.patch.selector import PatchSelector
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.transformer.utils.stack import TransformerEncoderStack
from Emperor.transformer.utils.patch.options.base import PatchConfig
from Emperor.transformer.utils.feed_forward import FeedForwardConfig
from Emperor.experiments.utils.models import TransformerClassifierExperiment
from Emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingOptions
from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingSelector
from Emperor.transformer.utils.embedding.options.base import PositionalEmbeddingConfig
from Emperor.base.enums import ActivationOptions, BaseOptions, LayerNormPositionOptions
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
)


@dataclass
class VITModelConfig(ConfigBase):
    patch_config: "PatchConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    positional_embedding_config: "PositionalEmbeddingConfig | None" = field(
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


class VITModel(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim

        self.main_cfg: VITModelConfig = self._resolve_main_config(self.cfg, cfg)
        self.patch_config = self.main_cfg.patch_config
        self.embedding_config = self.main_cfg.positional_embedding_config
        self.encoder_config = self.main_cfg.encoder_config
        self.output_config = self.main_cfg.output_config

        self.patch = PatchSelector(self.patch_config).build()
        self.positional_embedding = PositionalEmbeddingSelector(
            self.embedding_config
        ).build()
        self.transformer = TransformerEncoderStack(self.encoder_config)
        self.output = LayerStack(self.output_config).build_model()

    def forward(
        self,
        tokens_tensor: Tensor,
    ) -> Tensor:
        X = self.patch(tokens_tensor)
        X = self.positional_embedding(X)
        X, loss = self.transformer(X)
        X = self.__select_class_tokens(X)
        X = self.output(X)
        return X

    def __select_class_tokens(self, X: Tensor) -> Tensor:
        return X[:, 0, :]


class VITExperimentOptions(BaseOptions):
    BASE_VIT = 0
    ADAPTIVE_VIT = 1
    CUSTOM_VIT = 2


class VITExperiment(Experiments):
    def __init__(
        self,
        mini_datasetset_flag: bool = False,
    ) -> None:
        super().__init__(mini_datasetset_flag)

    def _get_learning_rates(self) -> list:
        return [1e-3, 1e-2]

    def train_model(
        self,
        model_config_options: VITExperimentOptions = VITExperimentOptions.BASE_VIT,
    ):
        preset = VITExperimentPresets().get_config(model_config_options)
        self._set_model_config(preset)
        self._train_model(VITModel, print_parameter_count_flag=True)


class VITExperimentPresets:
    def get_config(
        self,
        model_config_options: VITExperimentOptions = VITExperimentOptions.BASE_VIT,
    ) -> "ModelConfig":
        match model_config_options:
            case VITExperimentOptions.BASE_VIT:
                return self.base_linear_transformer_config()
            case VITExperimentOptions.ADAPTIVE_VIT:
                return self.adaptive_linear_transformer_config()
            case VITExperimentOptions.CUSTOM_VIT:
                return self.custom_linear_transformer_config()
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `VITExperimentOptions`."
                )

    def base_linear_transformer_config(self) -> "ModelConfig":
        BATCH_SIZE = 64
        INPUT_DIM = 32
        EMBEDDING_DIM = 32
        OUTPUT_DIM = 10
        DROPOUT_PROBABILITY = 0.0
        ACTIVATION_FUNCTION = ActivationOptions.RELU

        OUTPUT_NUM_LAYERS = 1
        TRANSFORMER_NUM_LAYERS = 1

        ATTN_BIAS_FLAG = False
        ATTN_NUM_LAYERS = 1
        FF_BIAS_FLAG = True
        FF_NUM_LAYERS = 2

        OUTPUT_BIAS_FLAG = True
        OUTPUT_NUM_LAYERS = 1

        IMAGE_PATCH_SIZE = 4
        CLASS_TOKEN_LENGTH = 1
        IMAGE_HEIGHT = IMAGE_HEIGHT = 28
        KERNEL_SIZE = IMAGE_PATCH_SIZE
        PADDING_SIZE = 0
        DILATATION_SIZE = 0
        STRIDE = IMAGE_PATCH_SIZE
        H_OUT = (
            IMAGE_HEIGHT + 2 * PADDING_SIZE - DILATATION_SIZE * (KERNEL_SIZE - 1) - 1
        ) // STRIDE + 1
        W_OUT = (
            IMAGE_HEIGHT + 2 * PADDING_SIZE - DILATATION_SIZE * (KERNEL_SIZE - 1) - 1
        ) // STRIDE + 1
        SEQUENCE_LENGTH = H_OUT * W_OUT + 1
        NUM_EMBEDDINGS = SEQUENCE_LENGTH - CLASS_TOKEN_LENGTH

        return ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=EMBEDDING_DIM,
            output_dim=OUTPUT_DIM,
            override_config=VITModelConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.LINEAR,
                    embedding_dim=EMBEDDING_DIM,
                    num_input_channels=1,
                    patch_size=IMAGE_PATCH_SIZE,
                    stride=IMAGE_PATCH_SIZE,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=EMBEDDING_DIM,
                        hidden_dim=EMBEDDING_DIM,
                        output_dim=EMBEDDING_DIM,
                        num_layers=1,
                        activation=ACTIVATION_FUNCTION,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=DROPOUT_PROBABILITY,
                        override_config=LinearLayerConfig(
                            input_dim=EMBEDDING_DIM,
                            output_dim=EMBEDDING_DIM,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=PositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=PositionalEmbeddingOptions.LEARNED,
                    num_embeddings=NUM_EMBEDDINGS,
                    embedding_dim=EMBEDDING_DIM,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerPresets.transformer_linear_base_preset(
                    batch_size=BATCH_SIZE,
                    num_layers=TRANSFORMER_NUM_LAYERS,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    num_heads=4,
                    embedding_dim=EMBEDDING_DIM,
                    query_key_projection_dim=EMBEDDING_DIM,
                    value_projection_dim=EMBEDDING_DIM,
                    target_sequence_length=SEQUENCE_LENGTH,
                    source_sequence_length=SEQUENCE_LENGTH,
                    target_dtype=torch.float32,
                    is_self_attention_projector_flag=True,
                    dropout_probability=DROPOUT_PROBABILITY,
                    key_value_bias_flag=False,
                    zero_attention_flag=False,
                    causal_attention_mask_flag=False,
                    add_key_value_bias_flag=False,
                    average_attention_weights_flag=False,
                    return_attention_weights_flag=False,
                    attn_stack_num_layers=ATTN_NUM_LAYERS,
                    attn_bias_flag=ATTN_BIAS_FLAG,
                    ff_stack_num_layers=FF_NUM_LAYERS,
                    ff_bias_flag=FF_BIAS_FLAG,
                    stack_activation=ACTIVATION_FUNCTION,
                    stack_residual_flag=False,
                ),
                output_config=LinearPresets.base_linear_layer_stack_preset(
                    batch_size=BATCH_SIZE,
                    input_dim=EMBEDDING_DIM,
                    output_dim=OUTPUT_DIM,
                    bias_flag=OUTPUT_BIAS_FLAG,
                    data_monitor=None,
                    parameter_monitor=None,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    stack_num_layers=OUTPUT_NUM_LAYERS,
                    stack_hidden_dim=EMBEDDING_DIM,
                    stack_activation=ACTIVATION_FUNCTION,
                    stack_residual_flag=False,
                    stack_dropout_probability=DROPOUT_PROBABILITY,
                ),
            ),
        )

    def adaptive_linear_transformer_config(self) -> "ModelConfig":
        BATCH_SIZE = 16
        INPUT_DIM = 32
        EMBEDDING_DIM = 32
        OUTPUT_DIM = 10
        DROPOUT_PROBABILITY = 0.1
        IMAGE_PATCH_SIZE = 4
        ACTIVATION_FUNCTION = ActivationOptions.GELU

        TRANSFORMER_NUM_LAYERS = 2

        ATTN_BIAS_FLAG = False
        ATTN_NUM_LAYERS = 1
        ATTN_GENERATOR_DEPTH = DynamicDepthOptions.DEPTH_OF_TWO
        ATTN_DIAGONAL_OPTION = DynamicDiagonalOptions.DIAGONAL
        ATTN_BIAS_OPTION = DynamicBiasOptions.DISABLED
        ATTN_BEHAVIOUR_STACK_NUM_LAYERS = 2

        FF_BIAS_FLAG = True
        FF_NUM_LAYERS = 2
        FF_GENERATOR_DEPTH = DynamicDepthOptions.DEPTH_OF_TWO
        FF_DIAGONAL_OPTION = DynamicDiagonalOptions.DIAGONAL
        FF_BIAS_OPTION = DynamicBiasOptions.DYNAMIC_PARAMETERS
        FF_BEHAVIOUR_STACK_NUM_LAYERS = 2

        OUTPUT_BIAS_FLAG = True
        OUTPUT_NUM_LAYERS = 1
        OUTPUT_GENERATOR_DEPTH = DynamicDepthOptions.DEPTH_OF_TWO
        OUTPUT_DIAGONAL_OPTIONS = DynamicDiagonalOptions.DIAGONAL
        OUTPUT_DYNAMIC_BIAS_OPTIONS = DynamicBiasOptions.DYNAMIC_PARAMETERS
        OUTPUT_BEHAVIOUR_STACK_NUM_LAYERS = 1

        IMAGE_PATCH_SIZE = 4
        CLASS_TOKEN_LENGTH = 1
        IMAGE_HEIGHT = IMAGE_HEIGHT = 28
        KERNEL_SIZE = IMAGE_PATCH_SIZE
        PADDING_SIZE = 0
        DILATATION_SIZE = 0
        STRIDE = IMAGE_PATCH_SIZE
        H_OUT = (
            IMAGE_HEIGHT + 2 * PADDING_SIZE - DILATATION_SIZE * (KERNEL_SIZE - 1) - 1
        ) // STRIDE + 1
        W_OUT = (
            IMAGE_HEIGHT + 2 * PADDING_SIZE - DILATATION_SIZE * (KERNEL_SIZE - 1) - 1
        ) // STRIDE + 1
        SEQUENCE_LENGTH = H_OUT * W_OUT + 1
        NUM_EMBEDDINGS = SEQUENCE_LENGTH - CLASS_TOKEN_LENGTH

        return ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=EMBEDDING_DIM,
            output_dim=OUTPUT_DIM,
            override_config=VITModelConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.LINEAR,
                    num_input_channels=1,
                    embedding_dim=EMBEDDING_DIM,
                    patch_size=IMAGE_PATCH_SIZE,
                    stride=IMAGE_PATCH_SIZE,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=EMBEDDING_DIM,
                        hidden_dim=EMBEDDING_DIM,
                        output_dim=EMBEDDING_DIM,
                        num_layers=1,
                        activation=ACTIVATION_FUNCTION,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=DROPOUT_PROBABILITY,
                        override_config=LinearLayerConfig(
                            input_dim=EMBEDDING_DIM,
                            output_dim=EMBEDDING_DIM,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=PositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=PositionalEmbeddingOptions.LEARNED,
                    num_embeddings=NUM_EMBEDDINGS,
                    embedding_dim=EMBEDDING_DIM,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerPresets.transformer_linear_adaptive_preset(
                    batch_size=BATCH_SIZE,
                    num_layers=TRANSFORMER_NUM_LAYERS,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    num_heads=4,
                    embedding_dim=EMBEDDING_DIM,
                    query_key_projection_dim=EMBEDDING_DIM,
                    value_projection_dim=EMBEDDING_DIM,
                    target_sequence_length=SEQUENCE_LENGTH,
                    source_sequence_length=SEQUENCE_LENGTH,
                    target_dtype=torch.float32,
                    is_self_attention_projector_flag=False,
                    dropout_probability=DROPOUT_PROBABILITY,
                    key_value_bias_flag=False,
                    zero_attention_flag=False,
                    causal_attention_mask_flag=False,
                    add_key_value_bias_flag=False,
                    average_attention_weights_flag=False,
                    return_attention_weights_flag=False,
                    attn_stack_num_layers=ATTN_NUM_LAYERS,
                    attn_bias_flag=ATTN_BIAS_FLAG,
                    attn_generator_depth=ATTN_GENERATOR_DEPTH,
                    attn_diagonal_option=ATTN_DIAGONAL_OPTION,
                    attn_bias_option=ATTN_BIAS_OPTION,
                    attn_behaviour_stack_num_layers=ATTN_BEHAVIOUR_STACK_NUM_LAYERS,
                    ff_stack_num_layers=FF_NUM_LAYERS,
                    ff_bias_flag=FF_BIAS_FLAG,
                    ff_generator_depth=FF_GENERATOR_DEPTH,
                    ff_diagonal_option=FF_DIAGONAL_OPTION,
                    ff_bias_option=FF_BIAS_OPTION,
                    ff_behaviour_stack_num_layers=FF_BEHAVIOUR_STACK_NUM_LAYERS,
                    stack_activation=ACTIVATION_FUNCTION,
                    stack_residual_flag=False,
                ),
                output_config=LinearPresets.adaptive_linear_layer_stack_preset(
                    batch_size=BATCH_SIZE,
                    input_dim=EMBEDDING_DIM,
                    output_dim=OUTPUT_DIM,
                    bias_flag=OUTPUT_BIAS_FLAG,
                    generator_depth=OUTPUT_GENERATOR_DEPTH,
                    diagonal_option=OUTPUT_DIAGONAL_OPTIONS,
                    bias_option=OUTPUT_DYNAMIC_BIAS_OPTIONS,
                    stack_num_layers=OUTPUT_NUM_LAYERS,
                    stack_hidden_dim=EMBEDDING_DIM,
                    stack_activation=ACTIVATION_FUNCTION,
                    stack_residual_flag=False,
                    stack_dropout_probability=DROPOUT_PROBABILITY,
                    adaptive_behaviour_stack_num_layers=OUTPUT_BEHAVIOUR_STACK_NUM_LAYERS,
                ),
            ),
        )

    def custom_linear_transformer_config(self) -> "ModelConfig":
        BATCH_SIZE = 64
        INPUT_DIM = 64
        EMBEDDING_DIM = 64
        OUTPUT_DIM = 10
        DROPOUT_PROBABILITY = 0.1
        ACTIVATION_FUNCTION = ActivationOptions.RELU

        OUTPUT_NUM_LAYERS = 1
        TRANSFORMER_NUM_LAYERS = 6

        ATTN_BIAS_FLAG = False
        ATTN_NUM_HEADS = 4
        ATTN_MODEL_TYPE = LinearLayerStackOptions.BASE
        ATTN_NUM_LAYERS = 1

        FF_BIAS_FLAG = True
        FF_MODEL_TYPE = LinearLayerStackOptions.BASE
        FF_NUM_LAYERS = 2

        OUTPUT_BIAS_FLAG = True
        OUTPUT_NUM_LAYERS = 2

        IMAGE_PATCH_SIZE = 4
        CLASS_TOKEN_LENGTH = 1
        IMAGE_HEIGHT = IMAGE_HEIGHT = 28
        KERNEL_SIZE = IMAGE_PATCH_SIZE
        PADDING_SIZE = 0
        DILATATION_SIZE = 0
        STRIDE = IMAGE_PATCH_SIZE
        H_OUT = (
            IMAGE_HEIGHT + 2 * PADDING_SIZE - DILATATION_SIZE * (KERNEL_SIZE - 1) - 1
        ) // STRIDE + 1
        W_OUT = (
            IMAGE_HEIGHT + 2 * PADDING_SIZE - DILATATION_SIZE * (KERNEL_SIZE - 1) - 1
        ) // STRIDE + 1
        SEQUENCE_LENGTH = H_OUT * W_OUT + 1
        NUM_EMBEDDINGS = SEQUENCE_LENGTH - CLASS_TOKEN_LENGTH

        return ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=EMBEDDING_DIM,
            output_dim=OUTPUT_DIM,
            override_config=VITModelConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.LINEAR,
                    embedding_dim=EMBEDDING_DIM,
                    num_input_channels=1,
                    patch_size=IMAGE_PATCH_SIZE,
                    stride=IMAGE_PATCH_SIZE,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=EMBEDDING_DIM,
                        hidden_dim=EMBEDDING_DIM,
                        output_dim=EMBEDDING_DIM,
                        num_layers=1,
                        activation=ACTIVATION_FUNCTION,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=DROPOUT_PROBABILITY,
                        override_config=LinearLayerConfig(
                            input_dim=EMBEDDING_DIM,
                            output_dim=EMBEDDING_DIM,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=PositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=PositionalEmbeddingOptions.LEARNED,
                    num_embeddings=NUM_EMBEDDINGS,
                    embedding_dim=EMBEDDING_DIM,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerConfig(
                    num_layers=TRANSFORMER_NUM_LAYERS,
                    source_sequence_length=SEQUENCE_LENGTH,
                    target_sequence_length=SEQUENCE_LENGTH,
                    embedding_dim=EMBEDDING_DIM,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    dropout_probability=DROPOUT_PROBABILITY,
                    causal_attention_mask_flag=False,
                    attention_config=MultiHeadAttentionConfig(
                        batch_size=BATCH_SIZE,
                        num_heads=ATTN_NUM_HEADS,
                        model_type=ATTN_MODEL_TYPE,
                        query_key_projection_dim=EMBEDDING_DIM,
                        value_projection_dim=EMBEDDING_DIM,
                        embedding_dim=EMBEDDING_DIM,
                        target_sequence_length=SEQUENCE_LENGTH,
                        source_sequence_length=SEQUENCE_LENGTH,
                        target_dtype=torch.float32,
                        is_self_attention_projector_flag=True,
                        dropout_probability=DROPOUT_PROBABILITY,
                        key_value_bias_flag=False,
                        zero_attention_flag=False,
                        causal_attention_mask_flag=False,
                        add_key_value_bias_flag=False,
                        average_attention_weights_flag=False,
                        return_attention_weights_flag=False,
                        override_config=LayerStackConfig(
                            input_dim=EMBEDDING_DIM,
                            hidden_dim=EMBEDDING_DIM,
                            output_dim=EMBEDDING_DIM,
                            num_layers=ATTN_NUM_LAYERS,
                            activation=ACTIVATION_FUNCTION,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=DROPOUT_PROBABILITY,
                            override_config=LinearLayerConfig(
                                input_dim=EMBEDDING_DIM,
                                output_dim=EMBEDDING_DIM,
                                bias_flag=ATTN_BIAS_FLAG,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                    feed_forward_config=FeedForwardConfig(
                        layer_stack_option=FF_MODEL_TYPE,
                        input_dim=EMBEDDING_DIM,
                        output_dim=EMBEDDING_DIM,
                        num_layers=FF_NUM_LAYERS,
                        override_config=LayerStackConfig(
                            input_dim=EMBEDDING_DIM,
                            hidden_dim=EMBEDDING_DIM,
                            output_dim=EMBEDDING_DIM,
                            num_layers=FF_NUM_LAYERS,
                            activation=ACTIVATION_FUNCTION,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=DROPOUT_PROBABILITY,
                            override_config=LinearLayerConfig(
                                input_dim=EMBEDDING_DIM,
                                output_dim=EMBEDDING_DIM,
                                bias_flag=FF_BIAS_FLAG,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                ),
                output_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=EMBEDDING_DIM,
                    hidden_dim=EMBEDDING_DIM,
                    output_dim=OUTPUT_DIM,
                    num_layers=OUTPUT_NUM_LAYERS,
                    activation=ACTIVATION_FUNCTION,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=False,
                    adaptive_computation_flag=False,
                    dropout_probability=DROPOUT_PROBABILITY,
                    override_config=LinearLayerConfig(
                        input_dim=EMBEDDING_DIM,
                        output_dim=OUTPUT_DIM,
                        bias_flag=OUTPUT_BIAS_FLAG,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    VITExperiment().train_model(VITExperimentOptions.CUSTOM_VIT)
