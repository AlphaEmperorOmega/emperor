import torch
import torch.nn as nn
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from Emperor.library.choice import Library as L
from Emperor.components.parameter_generators.utils.routers import RouterModel
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
import Emperor.components.parameter_generators.vector_choice as v
import Emperor.components.parameter_generators.matrix_choice as m
import Emperor.components.parameter_generators.generator_choice as g


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
class MixtureOfExpertsConfig(ModelConfig):
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
