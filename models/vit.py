import torch

from torch import Tensor
from models.parser import get_parser
from Emperor.config import ModelConfig
from dataclasses import dataclass, field
from Emperor.datasets.image.mnist import Mnist
from Emperor.base.utils import ConfigBase, Module
from Emperor.datasets.image.cifar_10 import Cifar10
from Emperor.datasets.image.cifar_100 import Cifar100
from Emperor.linears.utils.presets import LinearPresets
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.base.layer import LayerStack, LayerStackConfig
from Emperor.experiments.utils.factories import Experiments
from Emperor.datasets.image.fashion_mnist import FashionMNIST
from Emperor.transformer.utils.layers import TransformerConfig
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.transformer.utils.patch.selector import PatchOptions
from Emperor.transformer.utils.patch.selector import PatchSelector
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.transformer.utils.stack import TransformerEncoderStack
from Emperor.transformer.utils.patch.options.base import PatchConfig
from Emperor.transformer.utils.feed_forward import FeedForwardConfig
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
        X = tokens_tensor.to(self.device)
        X = self.patch(X)
        X = self.positional_embedding(X)
        X, loss = self.transformer(X)
        X = self.__select_class_tokens(X)
        X = self.output(X)
        return X

    def __select_class_tokens(self, X: Tensor) -> Tensor:
        return X[:, 0, :]


class VITExperimentOptions(BaseOptions):
    BASE = 0
    ADAPTIVE = 1


class VITExperiment(Experiments):
    def __init__(
        self,
        model_config_option: VITExperimentOptions | None = None,
        mini_datasetset_flag: bool = False,
    ) -> None:
        self.print_frequency = 50
        self.model_config_option = model_config_option
        super().__init__(mini_datasetset_flag, self.print_frequency)

    def _get_num_epochs(self) -> int:
        return 200

    def _get_learning_rates(self) -> list:
        return [1e-4, 1e-3, 1e-2]

    def _get_dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _get_model_config(self):
        if self.model_config_option is None:
            return None
        return VITExperimentPresets().get_config(self.model_config_option)

    def _get_model_type(self) -> type:
        return VITModel

    def train_model(self) -> None:
        if self.model_config_option is not None:
            super().train_model()
            return None

        for config_option in VITExperimentOptions:
            self.model_config_option = config_option
            config = VITExperimentPresets().get_config(config_option)
            self._set_model_config(config)
            super().train_model()


class VITExperimentPresets:
    def __init__(self) -> None:
        self.batch_size = 64

    def get_config(
        self,
        model_config_options: VITExperimentOptions = VITExperimentOptions.BASE,
    ) -> "ModelConfig":
        match model_config_options:
            case VITExperimentOptions.BASE:
                return self.__base_linear_transformer_config()
            case VITExperimentOptions.ADAPTIVE:
                return self.__adaptive_linear_transformer_config()
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `VITExperimentOptions`."
                )

    def __base_linear_transformer_config(self) -> "ModelConfig":
        self.input_dim = 32
        self.embedding_dim = 32
        self.output_dim = 10
        self.dropout_probability = 0.1
        self.activation_function = ActivationOptions.SILU

        self.output_num_layers = 1
        self.transformer_num_layers = 1

        self.attn_bias_flag = False
        self.attn_num_heads = 4
        self.attn_model_type = LinearLayerStackOptions.BASE
        self.attn_num_layers = 1

        self.ff_bias_flag = True
        self.ff_model_type = LinearLayerStackOptions.BASE
        self.ff_num_layers = 2

        self.output_bias_flag = True
        self.output_num_layers = 2

        self.image_patch_size = 4
        self.class_token_length = 1
        self.input_channels = 3
        # self.image_height = self.image_height = 28
        self.image_height = self.image_height = 32
        self.kernel_size = self.image_patch_size
        self.padding_size = 0
        self.dilatation_size = 0
        self.stride = self.image_patch_size
        self.h_out = (
            self.image_height
            + 2 * self.padding_size
            - self.dilatation_size * (self.kernel_size - 1)
            - 1
        ) // self.stride + 1
        self.w_out = (
            self.image_height
            + 2 * self.padding_size
            - self.dilatation_size * (self.kernel_size - 1)
            - 1
        ) // self.stride + 1
        self.sequence_length = self.h_out * self.w_out + 1
        self.num_embeddings = self.sequence_length - self.class_token_length

        return ModelConfig(
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.output_dim,
            override_config=VITModelConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.CONV,
                    embedding_dim=self.embedding_dim,
                    num_input_channels=self.input_channels,
                    patch_size=self.image_patch_size,
                    stride=self.image_patch_size,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=self.embedding_dim,
                        hidden_dim=self.embedding_dim,
                        output_dim=self.embedding_dim,
                        num_layers=1,
                        activation=self.activation_function,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=self.dropout_probability,
                        override_config=LinearLayerConfig(
                            input_dim=self.embedding_dim,
                            output_dim=self.embedding_dim,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=PositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=PositionalEmbeddingOptions.LEARNED,
                    num_embeddings=self.num_embeddings,
                    embedding_dim=self.embedding_dim,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerConfig(
                    num_layers=self.transformer_num_layers,
                    source_sequence_length=self.sequence_length,
                    target_sequence_length=self.sequence_length,
                    embedding_dim=self.embedding_dim,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    dropout_probability=self.dropout_probability,
                    causal_attention_mask_flag=False,
                    attention_config=MultiHeadAttentionConfig(
                        batch_size=self.batch_size,
                        num_heads=self.attn_num_heads,
                        model_type=self.attn_model_type,
                        query_key_projection_dim=self.embedding_dim,
                        value_projection_dim=self.embedding_dim,
                        embedding_dim=self.embedding_dim,
                        target_sequence_length=self.sequence_length,
                        source_sequence_length=self.sequence_length,
                        target_dtype=torch.float32,
                        is_self_attention_projector_flag=True,
                        dropout_probability=self.dropout_probability,
                        key_value_bias_flag=False,
                        zero_attention_flag=False,
                        causal_attention_mask_flag=False,
                        add_key_value_bias_flag=False,
                        average_attention_weights_flag=False,
                        return_attention_weights_flag=False,
                        override_config=LayerStackConfig(
                            input_dim=self.embedding_dim,
                            hidden_dim=self.embedding_dim,
                            output_dim=self.embedding_dim,
                            num_layers=self.attn_num_layers,
                            activation=self.activation_function,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=self.dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=self.embedding_dim,
                                output_dim=self.embedding_dim,
                                bias_flag=self.attn_bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                    feed_forward_config=FeedForwardConfig(
                        layer_stack_option=self.ff_model_type,
                        input_dim=self.embedding_dim,
                        output_dim=self.embedding_dim,
                        num_layers=self.ff_num_layers,
                        override_config=LayerStackConfig(
                            input_dim=self.embedding_dim,
                            hidden_dim=self.embedding_dim,
                            output_dim=self.embedding_dim,
                            num_layers=self.ff_num_layers,
                            activation=self.activation_function,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=self.dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=self.embedding_dim,
                                output_dim=self.embedding_dim,
                                bias_flag=self.ff_bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                ),
                output_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=self.embedding_dim,
                    hidden_dim=self.embedding_dim,
                    output_dim=self.output_dim,
                    num_layers=self.output_num_layers,
                    activation=self.activation_function,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=False,
                    adaptive_computation_flag=False,
                    dropout_probability=self.dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=self.embedding_dim,
                        output_dim=self.output_dim,
                        bias_flag=self.output_bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            ),
        )

    def __adaptive_linear_transformer_config(self) -> "ModelConfig":
        self.input_dim = 32
        self.embedding_dim = 32
        self.output_dim = 10
        self.dropout_probability = 0.1
        self.image_patch_size = 4
        self.activation_function = ActivationOptions.SILU

        self.transformer_num_layers = 1

        self.attn_bias_flag = True
        self.attn_num_layers = 1
        self.attn_generator_depth = DynamicDepthOptions.DEPTH_OF_TWO
        self.attn_diagonal_option = DynamicDiagonalOptions.DIAGONAL
        self.attn_bias_option = DynamicBiasOptions.DISABLED
        self.attn_behaviour_stack_num_layers = 2

        self.ff_bias_flag = True
        self.ff_num_layers = 2
        self.ff_generator_depth = DynamicDepthOptions.DEPTH_OF_TWO
        self.ff_diagonal_option = DynamicDiagonalOptions.DIAGONAL
        self.ff_bias_option = DynamicBiasOptions.DISABLED
        self.ff_behaviour_stack_num_layers = 2

        self.output_bias_flag = True
        self.output_num_layers = 1
        self.output_generator_depth = DynamicDepthOptions.DEPTH_OF_TWO
        self.output_diagonal_options = DynamicDiagonalOptions.DIAGONAL
        self.output_dynamic_bias_options = DynamicBiasOptions.DISABLED
        self.output_behaviour_stack_num_layers = 2

        # self.image_patch_size = 4
        # self.class_token_length = 1
        # self.image_height = 28

        self.image_patch_size = 4
        self.class_token_length = 1
        self.input_channels = 3
        self.image_height = self.image_height = 32
        self.kernel_size = self.image_patch_size
        self.padding_size = 0
        self.dilatation_size = 0
        self.stride = self.image_patch_size
        self.h_out = (
            self.image_height
            + 2 * self.padding_size
            - self.dilatation_size * (self.kernel_size - 1)
            - 1
        ) // self.stride + 1
        self.w_out = (
            self.image_height
            + 2 * self.padding_size
            - self.dilatation_size * (self.kernel_size - 1)
            - 1
        ) // self.stride + 1
        self.sequence_length = self.h_out * self.w_out + 1
        self.num_embeddings = self.sequence_length - self.class_token_length

        return ModelConfig(
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.output_dim,
            override_config=VITModelConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.CONV,
                    num_input_channels=self.input_channels,
                    embedding_dim=self.embedding_dim,
                    patch_size=self.image_patch_size,
                    stride=self.image_patch_size,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=self.embedding_dim,
                        hidden_dim=self.embedding_dim,
                        output_dim=self.embedding_dim,
                        num_layers=1,
                        activation=self.activation_function,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=self.dropout_probability,
                        override_config=LinearLayerConfig(
                            input_dim=self.embedding_dim,
                            output_dim=self.embedding_dim,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=PositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=PositionalEmbeddingOptions.LEARNED,
                    num_embeddings=self.num_embeddings,
                    embedding_dim=self.embedding_dim,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerPresets.transformer_linear_adaptive_preset(
                    batch_size=self.batch_size,
                    num_layers=self.transformer_num_layers,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    num_heads=4,
                    embedding_dim=self.embedding_dim,
                    query_key_projection_dim=self.embedding_dim,
                    value_projection_dim=self.embedding_dim,
                    target_sequence_length=self.sequence_length,
                    source_sequence_length=self.sequence_length,
                    target_dtype=torch.float32,
                    is_self_attention_projector_flag=False,
                    dropout_probability=self.dropout_probability,
                    key_value_bias_flag=False,
                    zero_attention_flag=False,
                    causal_attention_mask_flag=False,
                    add_key_value_bias_flag=False,
                    average_attention_weights_flag=False,
                    return_attention_weights_flag=False,
                    attn_stack_num_layers=self.attn_num_layers,
                    attn_bias_flag=self.attn_bias_flag,
                    attn_generator_depth=self.attn_generator_depth,
                    attn_diagonal_option=self.attn_diagonal_option,
                    attn_bias_option=self.attn_bias_option,
                    attn_behaviour_stack_num_layers=self.attn_behaviour_stack_num_layers,
                    ff_stack_num_layers=self.ff_num_layers,
                    ff_bias_flag=self.ff_bias_flag,
                    ff_generator_depth=self.ff_generator_depth,
                    ff_diagonal_option=self.ff_diagonal_option,
                    ff_bias_option=self.ff_bias_option,
                    ff_behaviour_stack_num_layers=self.ff_behaviour_stack_num_layers,
                    stack_activation=self.activation_function,
                    stack_residual_flag=False,
                ),
                output_config=LinearPresets.adaptive_linear_layer_stack_preset(
                    batch_size=self.batch_size,
                    input_dim=self.embedding_dim,
                    output_dim=self.output_dim,
                    bias_flag=self.output_bias_flag,
                    generator_depth=self.output_generator_depth,
                    diagonal_option=self.output_diagonal_options,
                    bias_option=self.output_dynamic_bias_options,
                    stack_num_layers=self.output_num_layers,
                    stack_hidden_dim=self.embedding_dim,
                    stack_activation=self.activation_function,
                    stack_residual_flag=False,
                    stack_dropout_probability=self.dropout_probability,
                    adaptive_behaviour_stack_num_layers=self.output_behaviour_stack_num_layers,
                ),
            ),
        )


if __name__ == "__main__":
    parser = get_parser(VITExperimentOptions.names())
    args = parser.parse_args()
    config_option = VITExperimentOptions.get_option(args.config_name)

    experiment = VITExperiment(config_option)
    experiment.train_model()
