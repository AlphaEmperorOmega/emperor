from dataclasses import dataclass

import models.bert.expert_linear.config as config
from models.bert.expert_linear.runtime_options import (
    BertEmbeddingOptions,
    BertMlmHeadOptions,
    BertNspHeadOptions,
)


@dataclass(frozen=True)
class BertBoundaryConfig:
    embedding_options: BertEmbeddingOptions
    mlm_head_options: BertMlmHeadOptions
    nsp_head_options: BertNspHeadOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    input_dim: int
    hidden_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: BertEmbeddingOptions | None
    mlm_head_options: BertMlmHeadOptions | None
    nsp_head_options: BertNspHeadOptions | None


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.input_dim = dependencies.input_dim
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.sequence_length = dependencies.sequence_length
        self.embedding_options = self.__default_embedding_options(
            dependencies.embedding_options
        )
        self.mlm_head_options = self.__default_mlm_head_options(
            dependencies.mlm_head_options
        )
        self.nsp_head_options = self.__default_nsp_head_options(
            dependencies.nsp_head_options
        )

    def build_boundary_config(self) -> BertBoundaryConfig:
        self.__validate()
        return BertBoundaryConfig(
            embedding_options=self.embedding_options,
            mlm_head_options=self.mlm_head_options,
            nsp_head_options=self.nsp_head_options,
        )

    def __default_embedding_options(
        self,
        embedding_options: BertEmbeddingOptions | None,
    ) -> BertEmbeddingOptions:
        if embedding_options is not None:
            return embedding_options
        return BertEmbeddingOptions(
            token_type_vocab_size=config.TOKEN_TYPE_VOCAB_SIZE,
            layer_norm_flag=config.EMBEDDING_LAYER_NORM_FLAG,
            dropout_probability=config.EMBEDDING_DROPOUT_PROBABILITY,
        )

    def __default_mlm_head_options(
        self,
        mlm_head_options: BertMlmHeadOptions | None,
    ) -> BertMlmHeadOptions:
        if mlm_head_options is not None:
            return mlm_head_options
        return BertMlmHeadOptions(
            activation=config.MLM_ACTIVATION,
            dense_bias_flag=config.MLM_DENSE_BIAS_FLAG,
            layer_norm_flag=config.MLM_LAYER_NORM_FLAG,
            decoder_bias_flag=config.MLM_DECODER_BIAS_FLAG,
            decoder_weight_tying_flag=config.MLM_DECODER_WEIGHT_TYING_FLAG,
        )

    def __default_nsp_head_options(
        self,
        nsp_head_options: BertNspHeadOptions | None,
    ) -> BertNspHeadOptions:
        if nsp_head_options is not None:
            return nsp_head_options
        return BertNspHeadOptions(
            pooler_activation=config.NSP_POOLER_ACTIVATION,
            pooler_bias_flag=config.NSP_POOLER_BIAS_FLAG,
            output_dim=config.NSP_OUTPUT_DIM,
            head_bias_flag=config.NSP_HEAD_BIAS_FLAG,
        )

    def __validate(self) -> None:
        self.__validate_positive_dimensions()
        self.__validate_embedding_dropout_probability()
        self.__validate_tied_vocabulary_dimensions()

    def __validate_positive_dimensions(self) -> None:
        dimensions = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "token_type_vocab_size": self.embedding_options.token_type_vocab_size,
            "nsp_output_dim": self.nsp_head_options.output_dim,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0, received {value}.")

    def __validate_embedding_dropout_probability(self) -> None:
        probability = self.embedding_options.dropout_probability
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "embedding dropout_probability must be in [0.0, 1.0], "
                f"received {probability}."
            )

    def __validate_tied_vocabulary_dimensions(self) -> None:
        if not self.mlm_head_options.decoder_weight_tying_flag:
            return
        if self.input_dim != self.output_dim:
            raise ValueError(
                "BERT MLM decoder weight tying requires input_dim to equal "
                f"output_dim, received input_dim={self.input_dim} and "
                f"output_dim={self.output_dim}."
            )
