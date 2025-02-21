import torch

import torch.nn as nn
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from traitlets import default

from Emperor.library.choice import Library as L
from Emperor.components.parameter_generators.utils.routers import RouterModel
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
import Emperor.components.parameter_generators.vector_choice as v
import Emperor.components.parameter_generators.matrix_choice as m
import Emperor.components.parameter_generators.generator_choice as g


NUM_EXPERTS: int = 12
HIDDEN_DIM: int = 12
TOPK: int = 12
ACTIVATION_FUNCTION: nn.Module = nn.SELU()
RANDOM_SAMPLE_TOPK: int = 3

# Auxiliary Losses
GENERATOR_AUXILIARY_LOSSES: Optional[AuxiliaryLosses] = None
MOE_AUXILIARY_LOSSES: Optional[AuxiliaryLosses] = None
CLUSTER_AUXILIARY_LOSSES: Optional[AuxiliaryLosses] = None

# Flags
MULTIPLY_BY_GATES: bool = True
ATTENTION_PROJECTION_BIAS_FLAG: bool = True
ADD_ZERO_ATTENTION_FLAG: bool = True
SELF_ATTENTION_FLAG: bool = True
ENCODER_DECODER_ATTENTION_FLAG: bool = False

# Transformer Attention Inputs
EMBEDDING_DIM: int = 784
QUERY_INPUT_DIM: Optional[int] = None
KEY_INPUT_DIM: Optional[int] = None
VALUE_INPUT_DIM: Optional[int] = None
QKV_HIDDEN_DIM: Optional[int] = 128
HEAD_DIM: int = 64  # Dimension an `attention head` has after the attention projection `QKV_HIDDEN_DIM` is split into multiple heads
ATTENTION_OUTPUT_DIM: Optional[int] = 10

# Transformer Config
QKV_INPUT_DIM: Optional[int] = None
QKV_OUTPUT_DIM: Optional[int] = None
FFN_INPUT_DIM: Optional[int] = None
FFN_HIDDEN_DIM: Optional[int] = None
FFN_OUTPUT_DIM: Optional[int] = None

# Additional Flags
RETURN_RAW_FFN_OUTPUT_FLAG: bool = False
NORMALIZE_BEFORE_FLAG: bool = False
ADD_MEMORY_BIAS_KEY_VALUES_FLAG: bool = False

# Dropout Probabilities
ATTN_DROPOUT_PROBABILITY: float = 0.0
FFN_DROPOUT_PROBABILITY: float = 0.0
DROPOUT_PROBABILITY: float = 0.0
QUANT_NOISE: float = 0.0
QUANT_BLOCK_SIZE: int = 0

# Gating
GATING_DROPOUT: int = 0  # Example: NUM_EXPERTS: int = 12

# AUXILIARY LOSSES
COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SWITCH_LOSS_WEIGHT: float = 0.0
ZERO_CENTERED_LOSS_WEIGHT: float = 0.0
MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0


@dataclass
class ParallelExpertsConfig:
    inputDim: int = field(default=EMBEDDING_DIM)
    hiddenDim: int = field(default=EMBEDDING_DIM)
    outputDim: int = field(default=EMBEDDING_DIM)
    multiplyByGatesFlag: bool = field(default=False)
    activationFunction: nn.Module = field(default=ACTIVATION_FUNCTION)


@dataclass
class MixtureOfExpertsConfig:
    inputDim: int = field(default=EMBEDDING_DIM)
    hiddenDim: int = field(default=EMBEDDING_DIM)
    outputDim: int = field(default=EMBEDDING_DIM)
    multiplyByGatesFlag: bool = field(default=False)
    activationFunction: nn.Module = field(default=ACTIVATION_FUNCTION)
    parallelExpertsConfig: ParallelExpertsConfig = ParameterGeneratorConfig()


@dataclass
class AttentionConfig:
    embeddingDim: Optional[int] = field(default=EMBEDDING_DIM)
    queryInputDim: Optional[int] = field(default=QUERY_INPUT_DIM)
    keyInputDim: Optional[int] = field(default=KEY_INPUT_DIM)
    valueInputDim: Optional[int] = field(default=VALUE_INPUT_DIM)
    qkvHiddenDim: Optional[int] = field(default=QKV_HIDDEN_DIM)
    attentionOutputDim: Optional[int] = field(default=ATTENTION_OUTPUT_DIM)
    dropoutProbability: Optional[float] = field(default=DROPOUT_PROBABILITY)
    biasFlag: Optional[bool] = field(default=ATTENTION_PROJECTION_BIAS_FLAG)
    addZeroAttentionFlag: Optional[bool] = field(default=ADD_ZERO_ATTENTION_FLAG)
    selfAttentionFlag: Optional[bool] = field(default=SELF_ATTENTION_FLAG)
    encoderDecoderAttentionFlag: Optional[bool] = field(
        default=ENCODER_DECODER_ATTENTION_FLAG
    )
    quantNoise: Optional[float] = field(default=QUANT_NOISE)
    quantBlockSize: Optional[int] = field(default=QUANT_BLOCK_SIZE)
    numExperts: Optional[int] = field(default=NUM_EXPERTS)
    topK: Optional[int] = field(default=TOPK)
    headDim: Optional[int] = field(default=HEAD_DIM)
    coefficientOfVariationLossWeight: Optional[float] = field(
        default=COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
    )
    switchLossWeight: Optional[float] = field(default=SWITCH_LOSS_WEIGHT)
    zeroCenteredLossWeight: Optional[float] = field(default=ZERO_CENTERED_LOSS_WEIGHT)
    mutualInformationLossWeight: Optional[float] = field(
        default=MUTUAL_INFORMATION_LOSS_WEIGHT
    )
    randomSampleTopK: Optional[int] = field(default=RANDOM_SAMPLE_TOPK)
    gatingDropout: Optional[int] = field(default=GATING_DROPOUT)
    addMemoryBiasKeyValuesFlag: Optional[bool] = field(
        default=ADD_MEMORY_BIAS_KEY_VALUES_FLAG
    )


@dataclass
class TransformerEncoderLayerConfig:
    embeddingDim: Optional[int] = field(default=EMBEDDING_DIM)
    returnRawFFNOutputFlag: Optional[bool] = field(default=RETURN_RAW_FFN_OUTPUT_FLAG)
    normalizeBeforeFlag: Optional[bool] = field(default=NORMALIZE_BEFORE_FLAG)
    activationFunction: Optional[nn.Module] = field(default=ACTIVATION_FUNCTION)
    attnDropoutProbability: Optional[float] = field(default=ATTN_DROPOUT_PROBABILITY)
    ffnDropoutProbability: Optional[float] = field(default=FFN_DROPOUT_PROBABILITY)
    attentionConfig: AttentionConfig = AttentionConfig()
    # mixtureOfExpertsConfig: AttentionConfig = AttentionConfig()


class ParameterGeneratorOptions(Enum):
    vector_choice_sparse = v.VectorChoiceSparse
    vector_choice_mixture = v.VectorChoiceMixture
    vector_choice_soft_mixture = v.VectorChoiceSoftMixture

    matrix_choice_sparse = m.MatrixChoiceSparse
    matrix_choice_mixture = m.MatrixChoiceMixture
    matrix_choice_soft_mixture = m.MatrixChoiceSoftMixture

    generator_sum = g.GeneratorChoiceSum
    generator_choice_mixture = g.GeneratorChoiceMixture
    generator_choice_soft_mixture = g.GeneratorChoiceSoftMixture

    def __str__(self):
        return self.value

    def create(self, cfg):
        return self.value(cfg)


@dataclass
class ParameterGeneratorConfig:
    auxiliaryLosses: Optional[AuxiliaryLosses] = field(default=None)
    generatorAuxiliaryLosses: Optional[AuxiliaryLosses] = field(default=None)
    moeAuxiliaryLosses: Optional[AuxiliaryLosses] = field(default=None)
    attentionAuxiliaryLosses: Optional[AuxiliaryLosses] = field(default=None)
    router: Optional[RouterModel] = field(default=None)
    plotProgress: bool = field(default=True)

    batchSize: int = field(default=5)
    sequenceLength: int = field(default=7)
    learningRate: float = field(default=0.1)
    inputDim: int = field(default=784)
    outputDim: int = field(default=10)
    depthDim: int = field(default=16)
    topK: int = field(default=4)
    parameterGeneartorType: ParameterGeneratorOptions = field(
        default=ParameterGeneratorOptions.vector_choice_sparse
    )

    noisyTopkFlag: bool = field(default=False)
    randomSampleTopK: int = field(default=0)

    mlpRouterFlag: bool = field(default=False)
    biasFlag: bool = field(default=True)

    # Features to be implemented
    gatherFrequencyFlag: bool = field(default=False)

    coefficientOfVariationLossWeight: float = field(default=0.1)
    switchLossWeight: float = field(default=0.1)
    zeroCentredLossWeight: float = field(default=0.0)
    mutualInformationLossWeight: float = field(default=0.0)

    # VectorChoiceEmbedding options
    # d_choice_embedding: int = field(default=64)


@dataclass
class ModelConfig(ParameterGeneratorConfig):
    numExperts: int = field(
        default=12,
        metadata={"help": "Number of experts for the `MixtureOfExperts` model"},
    )
    hiddenDim: int = field(
        default=12,
        metadata={
            "help": "Used as `input` and `output` dimension of a `intermediate` transformer layer"
        },
    )
    activationFunction: torch.nn = field(default=nn.SELU)
    generatorAuxiliaryLosses: Optional[AuxiliaryLosses] = field(default=None)
    moeAuxiliaryLosses: Optional[AuxiliaryLosses] = field(default=None)
    clusterAuxiliaryLosses: Optional[AuxiliaryLosses] = field(default=None)

    multiplyByGates: bool = field(default=True)
    attentionProjectionBiasFlag: bool = field(default=True)

    addZeroAttentionFlag: bool = field(default=True)
    selfAttentionFlag: bool = field(default=True)
    encoderDecorderAttentionFlag: bool = field(default=False)

    # TRANSFORMER ATTENTION INPUTS
    embeddingDim: int = field(default=784)
    queryInputDim: Optional[int] = field(default=None)
    keyInputDim: Optional[int] = field(default=None)
    valueInputDim: Optional[int] = field(default=None)
    qkvHiddenDim: Optional[int] = field(default=128)
    headDim: int = field(
        default=64,
        metadata={
            "help": "Dimension an `attention head` has after the attention projection `qkvHiddenDim` is split into multiple heads, where `numHeads = qkvHiddenDim // headDim`"
        },
    )
    attentionOutputDim: Optional[int] = field(default=10)

    # TRANSFOMER CONFIG
    qkvInputDim: Optional[int] = field(default=None)
    qkvOutputDim: Optional[int] = field(default=None)

    ffnInputDim: Optional[int] = field(default=None)
    ffnHiddenDim: Optional[int] = field(default=None)
    ffnOutputDim: Optional[int] = field(default=None)

    returnRawFFNOutputFlag: bool = field(default=False)
    normalizeBeforeFlag: bool = field(default=False)
    addMemoryBiasKeyValuesFlag: bool = field(default=False)
    attnDropoutProbability: float = field(default=0.0)
    ffnDropoutProbability: float = field(default=0.0)
    dropoutProbability: float = 0.0
    quantNoise: float = 0.0
    quantBlockSize: int = 0
    gatingDropout: int = 0

    def isNone(self, option):
        return option is None

    def setDefault(self, configOption, configOptionValue):
        if hasattr(self, configOption):
            if self.isNone(getattr(self, configOption)):
                setattr(self, configOption, configOptionValue)
        else:
            raise AttributeError(f"'{configOption}' is not a valid attribute")


@dataclass
class ParallelExpertsConfig(ModelConfig):
    numExperts_: Optional[int] = field(default=None)

    def __post_init__(self):
        self.setDefault("numExperts_", self.numExperts)


@dataclass
class MixtureOfExpertsConfigOld(ModelConfig):
    inputDim_: Optional[int] = field(default=None)
    hiddenDim_: Optional[int] = field(default=None)
    outputDim_: Optional[int] = field(default=None)
    numExperts_: Optional[int] = field(default=None)
    multiplyByGates_: Optional[bool] = field(default=False)

    def __post_init__(self):
        self.setDefault("inputDim_", self.inputDim)
        self.setDefault("hiddenDim_", self.hiddenDim)
        self.setDefault("outputDim_", self.outputDim)
        self.setDefault("numExperts_", self.numExperts)
        self.setDefault("multiplyByGates_", self.multiplyByGates)


@dataclass
class ParameterGeneratorMultiLayerConfig:
    hiddenDim: int = field(default=32)
    numberOfLayers: int = field(default=7)

    activatonFunction: torch.nn = field(default=nn.SELU)
    weightGeneratorLayerConfig: ParameterGeneratorConfig = field(
        default_factory=ParameterGeneratorConfig
    )
    firstWeightAndBiasGeneratorType: ParameterGeneratorOptions = field(
        default=ParameterGeneratorOptions.matrix_choice_sparse
    )
