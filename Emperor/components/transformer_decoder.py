import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Emperor.base.utils import Module
from Emperor.components.sut_layer import TransformerDecorderLayerBase
from Emperor.components.act import AdaptiveComputationTimeWrapper
from Emperor.config import ModelConfig

from typing import TYPE_CHECKING, Dict, Optional, List, Tuple, Any

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class TransformerDecoderBase(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        # TOOD: Temporary setting `dictionary` to an int
        dictionary: List,
        tokenEmbeddingModule,
        numLayers: Optional[List[int]] = None,
        maxSourceEmbeddingPositions: Optional[int] = None,
        useEncoderAttentionFlag: Optional[bool] = False,
        dictionaryDistributionModel: Optional[Module] = None,
        shareInputOutputEmbeddingFlag: Optional[bool] = None,
        inputEmbeddingDim: Optional[int] = None,
        outputEmbeddingDim: Optional[int] = None,
        crossSelfAttentionFlag: Optional[bool] = None,
        tieAdaptiveWeightsFlag: Optional[bool] = None,
        tokenEmbeddingWeightFlag: Optional[bool] = None,
        tokenEmbeddingDopoutProbability: Optional[float] = None,
        dynamicHaltingLossWeight: Optional[bool] = None,
        decoderHaltingFlag: Optional[bool] = None,
        tokenEmbeddingLayerNormFlag: Optional[bool] = None,
        addPositionalEmbeddingFlag: Optional[bool] = None,
        learnedPositionalEmbeddingFlag: Optional[bool] = None,
        normalizeBeforeFlag: Optional[bool] = None,
    ):
        super().__init__()
        self.register_buffer("version", torch.Tensor([1]))

        self.dictionary = dictionary

        self.tokenEmbeddingModule = tokenEmbeddingModule
        self.paddingIndex = self.tokenEmbeddingModule.padding_idx
        self.embeddingDim = self.tokenEmbeddingModule.embedding_dim

        self.cfg = cfg

        self.maxSourceEmbeddingPositions = self._getValue(
            maxSourceEmbeddingPositions, cfg.maxSourceEmbeddingPositions
        )
        self.tokenEmbeddingWeightFlag = self._getValue(
            tokenEmbeddingWeightFlag, cfg.tokenEmbeddingWeightFlag
        )
        self.shareInputOutputEmbeddingFlag = self._getValue(
            shareInputOutputEmbeddingFlag, cfg.shareInputOutputEmbeddingFlag
        )
        self.inputEmbeddingDim = self._getValue(
            inputEmbeddingDim, cfg.inputEmbeddingDim
        )
        self.outputEmbeddingDim = self._getValue(
            outputEmbeddingDim, cfg.outputEmbeddingDim
        )
        self.crossSelfAttentionFlag = self._getValue(
            crossSelfAttentionFlag, cfg.crossSelfAttentionFlag
        )
        self.tieAdaptiveWeightsFlag = self._getValue(
            tieAdaptiveWeightsFlag, cfg.tieAdaptiveWeightsFlag
        )
        self.numLayers = self._getValue(numLayers, cfg.numLayers)
        self.decoderHaltingFlag = self._getValue(
            decoderHaltingFlag, cfg.decoderHaltingFlag
        )
        self.dynamicHaltingLossWeight = self._getValue(
            dynamicHaltingLossWeight, cfg.dynamicHaltingLossWeight
        )
        self.tokenEmbeddingDopoutProbability = self._getValue(
            tokenEmbeddingDopoutProbability, cfg.tokenEmbeddingDopoutProbability
        )
        self.tokenEmbeddingLayerNormFlag = self._getValue(
            tokenEmbeddingLayerNormFlag, cfg.tokenEmbeddingLayerNormFlag
        )
        self.learnedPositionalEmbeddingFlag = self._getValue(
            learnedPositionalEmbeddingFlag, cfg.learnedPositionalEmbeddingFlag
        )
        self.normalizeBeforeFlag = self._getValue(
            normalizeBeforeFlag, cfg.normalizeBeforeFlag
        )

        # TODO: understand exaclty what this is used for
        self.futureMask = torch.empty(0)

        self.tokenEmbeddingWeight = (
            1.0 if self.tokenEmbeddingWeightFlag else math.sqrt(self.embeddingDim)
        )

        self.addPositionalEmbeddingFlag = self._getValue(
            addPositionalEmbeddingFlag, cfg.addPositionalEmbeddingFlag
        )

        self.dictionaryDistributionModel = dictionaryDistributionModel

        self.adaptive_softmax_cutoff = None

        self._initInputOutputProjectionModules()
        self._initTransformerLayer()
        self._initEmbeddingModules()
        self._initDictionaryModules()

    def _initDictionaryModules(self):
        self.adaptiveSoftmax = None
        if self.dictionaryDistributionModel is None:
            self.build_output_projection(
                self.cfg,
                self.dictionary,
                self.tokenEmbeddingModule,
            )

    def _initInputOutputProjectionModules(self):
        self.inputProjectionModel = None
        isInputEmbeddingDimEqual = self.embeddingDim != self.inputEmbeddingDim
        if isInputEmbeddingDimEqual:
            self.inputProjectionModel = nn.Linear(
                self.inputEmbeddingDim,
                self.embeddingDim,
                bias=False,
            )

        isOutputEmbeddingDimEqual = self.embeddingDim != self.outputEmbeddingDim
        tieEmbeddingDim = not self.tieAdaptiveWeightsFlag
        self.outputProectionModel = None
        if isOutputEmbeddingDimEqual and tieEmbeddingDim:
            self.outputProectionModel = nn.Linear(
                self.embeddingDim, self.outputEmbeddingDim, bias=False
            )

    def _initTransformerLayer(self):
        self.decoderModel = self._createTransformerModelHook()
        self.haltingFlag = False
        if self.decoderHaltingFlag:
            self.haltingFlag = True
            self.decoderModel = AdaptiveComputationTimeWrapper(
                self.cfg,
                self.decoderModel,
            )

    def _createTransformerModelHook(self):
        transformerLayer = TransformerDecorderLayerBase(
            self.cfg,
        )

        # TODO: At a later date figure out how to load different checkpoints
        # checkpoint = cfg.checkpoint_activations
        # if checkpoint:
        #     offload_to_cpu = cfg.offload_activations
        #     layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        # min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        # layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)

        return transformerLayer

    def _initEmbeddingModules(self):
        self.tokenEmbeddingDropoutModule = nn.Dropout(
            self.tokenEmbeddingDopoutProbability
        )

        if self.tokenEmbeddingLayerNormFlag:
            self.layerNormEmbeddingModule = nn.LayerNorm(self.embeddingDim)

        if self.addPositionalEmbeddingFlag:
            self.positionalEmbedding = PositionalEmbedding(
                self.maxSourceEmbeddingPositions,
                self.embeddingDim,
                self.paddingIndex,
                learned=self.learnedPositionalEmbeddingFlag,
            )

        self.quantNoiseFlag = False
        self.quantNoiseModule = None
        # TODO: figure out later how this works
        # if not cfg.adaptive_input and cfg.quantNoise.pq > 0:
        #     self.quantNoise = apply_quant_noise_(
        #         nn.Linear(embed_dim, embed_dim, bias=False),
        #         cfg.quantNoise.pq,
        #         cfg.quantNoise.pq_block_size,
        #     )

    def _initTransformerModules(self):
        if self.normalizeBeforeFlag:
            self.transformerLayerNorm = nn.LayerNorm(self.embeddingDim)

        # Not used for this model
        # self.layerDropProbability = cfg.encoder.layerdrop
        # if self.layerDropProbability > 0.0:
        #     self.layers = LayerDropModuleList(p=self.layerDropProbability)
        # else:
        #     self.layers = nn.ModuleList([])

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if self.adaptive_softmax_cutoff is not None:
            self.adaptiveSoftmax = AdaptiveSoftmax(
                len(dictionary),
                self.outputEmbeddingDim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.shareInputOutputEmbeddingFlag:
            self.dictionaryDistributionModel = nn.Linear(
                self.tokenEmbeddingModule.weight.shape[1],
                self.tokenEmbeddingModule.weight.shape[0],
                bias=False,
            )
            self.dictionaryDistributionModel.weight = self.tokenEmbeddingModule.weight
        else:
            self.dictionaryDistributionModel = nn.Linear(
                self.outputEmbeddingDim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.dictionaryDistributionModel.weight,
                mean=0,
                std=self.outputEmbeddingDim**-0.5,
            )

    def forward(
        self,
        targetTokens: Optional[Tensor] = None,
        tokenEmbeddings: Optional[Tensor] = None,
        encoderOutputDict: Optional[Dict[str, List[Tensor]]] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        featuresOnlyFlag: bool = False,
        fullContextAlignmentFlag: bool = False,
        alignmentLayer: Optional[int] = None,
        alignmentHeads: Optional[int] = None,
    ):
        assert targetTokens is not None or tokenEmbeddings is not None

        if alignmentLayer is None:
            alignmentLayer = self.numLayers - 1

        encoderOutputTensor, encoderPaddingMask = self._getEncoderOutputAndPaddingMask(
            targetTokens,
            tokenEmbeddings,
            encoderOutputDict,
        )

        tokenEmbeddings = self._computeTokenEmbeddings(
            targetTokens, tokenEmbeddings, incrementalState
        )

        (
            layerOutput,
            attentionWeights,
            adaptiveComputationTimeAttention,
            adaptiveComputationTimeLoss,
            layerHiddenStates,
        ) = self._computeAllLayersOutput(
            targetTokens=targetTokens,
            tokenEmbeddings=tokenEmbeddings,
            encoderOutputTensor=encoderOutputTensor,
            encoderPaddingMask=encoderPaddingMask,
            incrementalState=incrementalState,
            alignmentLayer=alignmentLayer,
            fullContextAlignmentFlag=fullContextAlignmentFlag,
        )

        layerOutput, results = self._prepareOutput(
            layersOutput=layerOutput,
            attentionWeights=attentionWeights,
            adaptiveComputationTimeAttention=adaptiveComputationTimeAttention,
            adaptiveComputationTimeLoss=adaptiveComputationTimeLoss,
            layerHiddenStates=layerHiddenStates,
            encoderOutputDict=encoderOutputDict,
            alignmentHeads=alignmentHeads,
        )

        if not featuresOnlyFlag:
            layerOutput = self._outputLayer(layerOutput)
        return layerOutput, results

    def _getEncoderOutputAndPaddingMask(
        self,
        targetTokens: Optional[Tensor] = None,
        tokenEmbeddings: Optional[Tensor] = None,
        encoderOutput: Optional[Dict[str, List[Tensor]]] = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        assert targetTokens is not None or tokenEmbeddings is not None, (
            f"Expected either `targetTokens` or `tokenEmbeddings` to be tensors, received `targetTokens`: {type(targetTokens)} and `tokenEmbeddings`: {type(tokenEmbeddings)}"
        )
        encoderOutputTensor: Optional[Tensor] = None
        encoderPaddingMask: Optional[Tensor] = None
        if targetTokens is not None:
            decoderBatchSize, _ = targetTokens.size()
        else:
            decoderBatchSize, _, _ = tokenEmbeddings.size()
        if encoderOutput is not None and len(encoderOutput["encoderOutput"]) > 0:
            encoderOutputTensor = encoderOutput["encoderOutput"][0]
            _, encoderBatchSize, _ = encoderOutputTensor.size()
            assert encoderBatchSize == decoderBatchSize, (
                f"The `batchSize` encoder: {encoderBatchSize} and decoder {decoderBatchSize} is expected to be the same."
            )
            if len(encoderOutput["encoderPaddingMask"]) > 0:
                encoderPaddingMask = encoderOutput["encoderPaddingMask"][0]

        return encoderOutputTensor, encoderPaddingMask

    def _computeTokenEmbeddings(
        self,
        targetTokens: Optional[Tensor] = None,
        tokenEmbeddings: Optional[torch.Tensor] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        isIncrementalState = incrementalState is not None
        if isIncrementalState:
            targetTokens = targetTokens[:, -1:]

        tokenEmbeddings = self._retrieveTokenEmbedding(
            targetTokens,
            tokenEmbeddings,
            isIncrementalState,
        )

        # B x T x C -> T x B x C
        tokenEmbeddings = tokenEmbeddings.transpose(0, 1)

        return tokenEmbeddings

    def _retrieveTokenEmbedding(
        self,
        targetTokens: Optional[Tensor] = None,
        tokenEmbeddings: Optional[Tensor] = None,
        isIncrementalStateFlag: bool = False,
    ) -> Tensor:
        if tokenEmbeddings is None:
            tokenEmbeddings = self.tokenEmbeddingModule(targetTokens)

        x = self.tokenEmbeddingWeight * tokenEmbeddings

        # TODO: Implement the quant nosie in `TransformerDecoderBase`
        # if self.quantNoiseFlag is not None:
        #     x = self.quantNoiseModule(x)

        if self.inputProjectionModel is not None:
            x = self.inputProjectionModel(x)

        positinalEmbedding = None
        # TODO: Find a way to apply positional encoding to image tokenEmbeddings
        # because at the moment `targetTokens` are required to compute `positinalEmbedding`
        if self.addPositionalEmbeddingFlag and targetTokens is not None:
            positinalEmbedding = self.positionalEmbedding(targetTokens)
            if isIncrementalStateFlag:
                positinalEmbedding = positinalEmbedding[:, -1:]
            x += positinalEmbedding

        if self.tokenEmbeddingLayerNormFlag:
            x = self.layerNormEmbeddingModule(x)

        x = self.tokenEmbeddingDropoutModule(x)

        return x

    def _computeAllLayersOutput(
        self,
        targetTokens: Optional[Tensor],
        tokenEmbeddings: Optional[Tensor],
        encoderOutputTensor: Optional[Tensor],
        encoderPaddingMask: Optional[Tensor],
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        alignmentLayer: Optional[int] = None,
        fullContextAlignmentFlag: bool = False,
    ):
        adaptiveComputationTimeLoss = 0.0
        layerHiddenStates = [tokenEmbeddings]
        attentionWeights = adaptieComputationTimeAttention = (
            adaptiveComputationTimeState
        ) = None
        selfAttentionPaddingMask = self._computeSelfAttentionMask(targetTokens)
        haltMask = self._computeHaltingMask(targetTokens)

        layerOutput = tokenEmbeddings
        for layerIdx in range(self.numLayers):
            (
                layerOutput,
                adaptieComputationTimeAttention,
                adaptiveComputationTimeState,
                adaptiveComputationTimeLoss,
            ) = self._computeLayerOutput(
                tokenEmbeddings=layerOutput,
                selfAttentionInput=adaptieComputationTimeAttention,
                haltMask=haltMask,
                layerIdx=layerIdx,
                encoderOutputTensor=encoderOutputTensor,
                encoderPaddingMask=encoderPaddingMask,
                selfAttentionPaddingMask=selfAttentionPaddingMask,
                incrementalState=incrementalState,
                adaptiveComputationTimeState=adaptiveComputationTimeState,
                fullContextAlignmentFlag=fullContextAlignmentFlag,
            )

            layerOutput, attentionWeightsTensor, _ = layerOutput

            # TODO: Find out why `attentionWeightsTensor` has to be moved to
            # the same GPU as layer output when `layerOutput`
            attentionWeights = None
            if attentionWeightsTensor is not None and layerIdx == alignmentLayer:
                attentionWeights = attentionWeightsTensor.float().to(layerOutput)

        return (
            layerOutput,
            attentionWeights,
            adaptieComputationTimeAttention,
            adaptiveComputationTimeLoss,
            layerHiddenStates,
        )

    def _computeLayerOutput(
        self,
        tokenEmbeddings: Tensor,
        selfAttentionInput: Optional[Tensor],
        haltMask: Optional[Tensor],
        layerIdx: Optional[int],
        encoderOutputTensor: Optional[Tensor],
        encoderPaddingMask: Optional[Tensor],
        selfAttentionPaddingMask: Optional[Tensor],
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        adaptiveComputationTimeState: Optional[Tuple] = None,
        fullContextAlignmentFlag: bool = False,
    ):
        selfAttentionMask = None
        if incrementalState is None and not fullContextAlignmentFlag:
            selfAttentionMask = self._bufferedFutureMask(tokenEmbeddings)

        if self.haltingFlag:
            (
                adaptiveComputationTimeState,
                layerOutput,
                selfAttentionInput,
                adaptiveComputationTimeLoss,
            ) = self.decoderModel.forward(
                previousAdaptiveComputationState=adaptiveComputationTimeState,
                previousHiddenState=tokenEmbeddings,
                selfAttentionInput=selfAttentionInput,
                paddingMask=haltMask,
                layerIdx=layerIdx,
                encoderOutput=encoderOutputTensor,
                encoderPaddingMask=encoderPaddingMask,
                incrementalState=incrementalState,
                selfAttentionMask=selfAttentionMask,
                selfAttentionPaddingMask=selfAttentionPaddingMask,
                # need_attn=bool((idx == alignment_layer)),
                # needHeadWeightsFlag=bool((idx == alignment_layer)),
            )

            return (
                layerOutput,
                selfAttentionInput,
                adaptiveComputationTimeState,
                adaptiveComputationTimeLoss,
            )

        else:
            layerOutput = self.decoderModel.forward(
                inputBatch=tokenEmbeddings,
                selfAttentionInput=selfAttentionInput,
                haltMask=haltMask,
                layerIdx=layerIdx,
                encoderOutput=encoderOutputTensor,
                encoderPaddingMask=encoderPaddingMask,
                incrementalState=incrementalState,
                selfAttentionMask=selfAttentionMask,
                selfAttentionPaddingMask=selfAttentionPaddingMask,
                # needHeadWeightsFlag=bool((layerIdx == alignmentLayer)),
            )

            return layerOutput, None, None, 0.0

    def _computeSelfAttentionMask(
        self,
        targetTokens: Optional[Tensor] = None,
    ):
        selfAttentionPaddingMask: Optional[Tensor] = None
        if self.crossSelfAttentionFlag:
            if targetTokens.eq(self.paddingIndex).any():
                selfAttentionPaddingMask = targetTokens.eq(self.paddingIndex)
        return selfAttentionPaddingMask

    def _computeHaltingMask(
        self,
        targetTokens: Optional[Tensor] = None,
    ):
        if targetTokens is not None:
            haltBooleanMask = targetTokens.eq(self.paddingIndex)
        else:
            haltBooleanMask = torch.zeros(
                self.cfg.batchSize, self.cfg.sequenceLength, dtype=torch.bool
            )
        haltMask = (1.0 - haltBooleanMask.t().float()).contiguous()
        return haltMask

    def _prepareOutput(
        self,
        layersOutput: Tensor,
        attentionWeights: Tensor,
        adaptiveComputationTimeAttention: Tensor,
        adaptiveComputationTimeLoss: Tensor,
        layerHiddenStates: List,
        encoderOutputDict: Optional[Dict[str, List[Tensor]]],
        alignmentHeads: int,
    ):
        totalDecoderLoss = 0.0
        if self.haltingFlag:
            # TODO: Think about why softhalting is replacing the
            # layer output when halting is used
            layersOutput = adaptiveComputationTimeAttention
            totalDecoderLoss = (
                self.dynamicHaltingLossWeight * adaptiveComputationTimeLoss
            )

        if attentionWeights is not None:
            if alignmentHeads is not None:
                attentionWeights = attentionWeights[:alignmentHeads]

            attentionWeights = attentionWeights.mean(dim=0)

        if self.normalizeBeforeFlag:
            layersOutput = self.transformerLayerNorm(layersOutput)

        # T x B x C -> B x T x C
        layersOutput = layersOutput.transpose(0, 1)

        if self.outputProectionModel is not None:
            layersOutput = self.outputProectionModel(layersOutput)

        if encoderOutputDict is not None:
            # WARNING: the encoder output is required here it'S nto optional
            # meaning this is meant to be a encoder-decoder transformer
            results = {
                "attentionWeights": [attentionWeights],
                "layerHiddenStates": layerHiddenStates,
                "totalDecoderLoss": encoderOutputDict["encoderLoss"][0]
                + totalDecoderLoss,
            }

            if "encoderHaltLoss" in encoderOutputDict:
                results["decoderHatlLoss"] = adaptiveComputationTimeLoss
                results["encoderHaltLoss"] = encoderOutputDict["encoderHaltLoss"]

            return layersOutput, results
        return layersOutput, None

    def _outputLayer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptiveSoftmax is None:
            return self.dictionaryDistributionModel(features)
        return features

    def _bufferedFutureMask(self, tensor):
        dim = tensor.size(0)
        if (
            self.futureMask.size(0) == 0
            or (not self.futureMask.device == tensor.device)
            or self.futureMask.size(0) < dim
        ):
            self.futureMask = torch.triu(
                self._fillWithNegativeInfinity(torch.zeros([dim, dim])), 1
            )
        self.futureMask = self.futureMask.to(tensor)
        return self.futureMask[:dim, :dim]

    def _fillWithNegativeInfinity(self, tensor):
        return tensor.float().fill_(float("-inf")).type_as(tensor)


def checkpoint_wrapper(m, offload_to_cpu=False):
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:
    - wraps an nn.Module, so that all subsequent calls will use checkpointing
    - handles keyword arguments in the forward
    - handles non-Tensor outputs from the forward

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    """
    # should I check whether original_forward has already been set?
    assert not hasattr(m, "precheckpoint_forward"), (
        "checkpoint function has already been applied?"
    )
    m.precheckpoint_forward = m.forward
    m.forward = functools.partial(
        _checkpointed_forward,
        m.precheckpoint_forward,  # original_forward
        offload_to_cpu,
    )
    return m


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
    auto_expand: bool = True,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
            auto_expand=auto_expand,
        )
    return m


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (self.padding_idx is None), (
            "If positions is pre-computed then padding_idx should not be set."
        )

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1024, auto_expand=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer(
            "weights",
            SinusoidalPositionalEmbedding.get_embedding(
                init_size, embedding_dim, padding_idx
            ),
            persistent=False,
        )
        self.max_positions = int(1e5)
        self.auto_expand = auto_expand
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ignore some deprecated keys that were used in older versions
        deprecated_keys = ["weights", "_float_tensor"]
        for key in deprecated_keys:
            if prefix + key in state_dict:
                del state_dict[prefix + key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        weights = self.weights

        if max_pos > self.weights.size(0):
            # If the input is longer than the number of pre-computed embeddings,
            # compute the extra embeddings on the fly.
            # Only store the expanded embeddings if auto_expand=True.
            # In multithreading environments, mutating the weights of a module
            # may cause trouble. Set auto_expand=False if this happens.
            weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(self.weights)
            if self.auto_expand:
                self.weights = weights

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
        )
