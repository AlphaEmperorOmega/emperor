import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, dtype, export
from torch.nn.modules import transformer
from Emperor.base.utils import Module
from Emperor.components.sut_layer import TransformerEncoderLayerBase
from Emperor.components.act import AdaptiveComputationTimeWrapper
from Emperor.config import ModelConfig

from typing import TYPE_CHECKING, Dict, Optional, List, Tuple, Any

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class TransformerEncoderBase(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        # dictionary,
        tokenEmbeddingModule,
        numLayers: Optional[int] = None,
        maxSourceEmbeddingPositions: Optional[int] = None,
        dynamicHaltingLossWeight: Optional[float] = None,
        tokenEmbeddingDopoutProbability: Optional[float] = None,
        tokenEmbeddingLayerNormFlag: Optional[float] = None,
        tokenEmbeddingWeightFlag: Optional[float] = None,
        addPositionalEmbeddingFlag: Optional[float] = None,
        normalizeBeforeFlag: Optional[bool] = None,
        quantNoiseFlag: Optional[bool] = None,
        gatherVusalizationDataFlag: Optional[bool] = None,
        returnAllHiddensFlag: Optional[bool] = None,
        encoderHaltingFlag: Optional[bool] = None,
        learnedPositionalEmbeddingFlag: Optional[bool] = None,
    ):
        super().__init__()
        self.register_buffer("version", torch.Tensor([1]))

        self.tokenEmbeddingModule = tokenEmbeddingModule
        self.paddingIndex = self.tokenEmbeddingModule.padding_idx
        self.embeddingDim = self.tokenEmbeddingModule.embedding_dim

        self.cfg = cfg
        self.maxSourceEmbeddingPositions = self._getValue(
            maxSourceEmbeddingPositions, cfg.maxSourceEmbeddingPositions
        )
        self.numLayers = self._getValue(numLayers, cfg.numLayers)
        self.dynamicHaltingLossWeight = self._getValue(
            dynamicHaltingLossWeight, cfg.dynamicHaltingLossWeight
        )
        self.tokenEmbeddingDopoutProbability = self._getValue(
            tokenEmbeddingDopoutProbability, cfg.tokenEmbeddingDopoutProbability
        )
        self.tokenEmbeddingLayerNormFlag = self._getValue(
            tokenEmbeddingLayerNormFlag, cfg.tokenEmbeddingLayerNormFlag
        )
        self.addPositionalEmbeddingFlag = self._getValue(
            addPositionalEmbeddingFlag, cfg.addPositionalEmbeddingFlag
        )
        self.normalizeBeforeFlag = self._getValue(
            normalizeBeforeFlag, cfg.normalizeBeforeFlag
        )
        self.tokenEmbeddingWeightFlag = self._getValue(
            tokenEmbeddingWeightFlag, cfg.tokenEmbeddingWeightFlag
        )
        self.quantNoiseFlag = self._getValue(quantNoiseFlag, cfg.quantNoiseFlag)

        self.gatherVusalizationDataFlag = self._getValue(
            gatherVusalizationDataFlag, cfg.gatherVusalizationDataFlag
        )
        self.returnAllHiddensFlag = self._getValue(
            returnAllHiddensFlag, cfg.returnAllHiddensFlag
        )
        self.encoderHaltingFlag = self._getValue(
            encoderHaltingFlag, cfg.encoderHaltingFlag
        )

        self.learnedPositionalEmbeddingFlag = self._getValue(
            learnedPositionalEmbeddingFlag, cfg.learnedPositionalEmbeddingFlag
        )

        self.tokenEmbeddingWeight = (
            1.0 if self.tokenEmbeddingWeightFlag else math.sqrt(self.embeddingDim)
        )

        self._resetDataGathering()
        self._initTransformerModules()
        self._initTransformerLayer()
        self._initEmbeddingModules()

    def _resetDataGathering(self):
        self.encoderStatesList = []
        self.ffnRawOutputList = []
        self.haltingMaksLogitsList = []
        self.haltedMasksList = []
        self.routeIndexesList = []

    def _initTransformerModules(self):
        if self.normalizeBeforeFlag:
            self.transformerLayerNorm = nn.LayerNorm(self.embeddingDim)

        # Not used for this model
        # self.layerDropProbability = cfg.encoder.layerdrop
        # if self.layerDropProbability > 0.0:
        #     self.layers = LayerDropModuleList(p=self.layerDropProbability)
        # else:
        #     self.layers = nn.ModuleList([])

    def _initTransformerLayer(self):
        self.encoderModel = self._createTransformerModelHook()

        self.haltingFlag = False
        if self.encoderHaltingFlag:
            self.haltingFlag = True
            self.encoderModel = AdaptiveComputationTimeWrapper(
                self.cfg,
                self.encoderModel,
            )

    def _createTransformerModelHook(self):
        transformerLayer = TransformerEncoderLayerBase(
            self.cfg,
        )

        # TODO: Come back later when figure out how checkpoints work
        # checkpoint = cfg.checkpoint_activations
        # if checkpoint:
        #     offload_to_cpu = cfg.offload_activations
        #     layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # # if we are checkpointing, enforce that FSDP always wraps the
        # # checkpointed layer, regardless of layer size
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

    def forward(
        self,
        sourceTokens: Optional[Tensor] = None,
        sourceSequenceLengths: Optional[torch.Tensor] = None,
        tokenEmbeddings: Optional[torch.Tensor] = None,
    ):
        self._resetDataGathering()
        tokenEmbeddings, paddingMask, rawTokenEmbedding, hasPaddingMask = (
            self._computeTokenEmbeddings(sourceTokens, tokenEmbeddings)
        )

        layersOutput, softHaltingInput, adaptiveComputationTimeLoss = (
            self._computeAllLayersOutput(
                tokenEmbeddings=tokenEmbeddings,
                paddingMask=paddingMask,
                hasPaddingMask=hasPaddingMask,
            )
        )

        layersOutput, totalEncoderLoss, sourceSequenceLengths = self._prepareOutput(
            layersOutput=layersOutput,
            sourceTokens=sourceTokens,
            softHaltingInput=softHaltingInput,
            adaptiveComputationTimeLoss=adaptiveComputationTimeLoss,
        )

        self._storeVusalisationData()

        return {
            "encoderOutput": [layersOutput],  # T x B x C
            "encoderPaddingMask": [paddingMask],  # B x T
            "encoderRawEmbeddings": [rawTokenEmbedding],  # B x T x C
            "encoderStates": self.encoderStatesList,  # List[T x B x C]
            "ffnRawOutputList": self.ffnRawOutputList,  # List[T x B x C]
            "sourceTokens": [],
            "sourceSequenceLengths": [sourceSequenceLengths],
            "encoderLoss": [totalEncoderLoss],
            "encoderHaltLoss": adaptiveComputationTimeLoss,
        }

    def _computeTokenEmbeddings(
        self,
        sourceTokens: Optional[Tensor] = None,
        tokenEmbeddings: Optional[torch.Tensor] = None,
    ):
        if sourceTokens is not None:
            paddingMask = sourceTokens.eq(self.paddingIndex)
            hasPaddingMask = sourceTokens.device.type == "xla" or paddingMask.any()

        if sourceTokens is None and tokenEmbeddings is not None:
            batchSize, sequenceLength, embeddingDim = tokenEmbeddings.size()
            paddingMask = torch.zeros((batchSize, sequenceLength), dtype=torch.bool)
            hasPaddingMask = False

        tokenEmbeddings, rawTokenEmbedding = self._retrieveTokenEmbedding(
            sourceTokens, tokenEmbeddings
        )

        tokenEmbeddings = self._applyPaddingMask(
            tokenEmbeddings, paddingMask, hasPaddingMask
        )

        # B x T x C -> T x B x C
        tokenEmbeddings = tokenEmbeddings.transpose(0, 1)

        if self.returnAllHiddensFlag:
            self.encoderStatesList.append(tokenEmbeddings)

        return tokenEmbeddings, paddingMask, rawTokenEmbedding, hasPaddingMask

    def _applyPaddingMask(
        self, tokenEmbeddings: Tensor, paddingMask: Tensor, hasPaddingMask: bool
    ):
        if hasPaddingMask:
            binaryPaddingMask = paddingMask.unsqueeze(-1).type_as(tokenEmbeddings)
            tokenEmbeddings = tokenEmbeddings * (1 - binaryPaddingMask)
        return tokenEmbeddings

    def _retrieveTokenEmbedding(
        self,
        sourceTokens: Optional[Tensor] = None,
        tokenEmbeddings: Optional[torch.Tensor] = None,
    ):
        if tokenEmbeddings is None:
            tokenEmbeddings = self.tokenEmbeddingModule(sourceTokens)

        x = rawTokenEmbedding = self.tokenEmbeddingWeight * tokenEmbeddings

        positionalEmbedding = None
        if self.addPositionalEmbeddingFlag and sourceTokens is not None:
            positionalEmbedding = self.positionalEmbedding(sourceTokens)
            x += positionalEmbedding

        if self.tokenEmbeddingLayerNormFlag:
            x = self.layerNormEmbeddingModule(x)

        x = self.tokenEmbeddingDropoutModule(x)

        # TODO: Implement the quant nosie in `TransformerEncoderBase`
        # if self.quantNoiseFlag:
        #     x = self.quantNoiseModule(x)

        return x, rawTokenEmbedding

    def _computeAllLayersOutput(
        self,
        tokenEmbeddings: Tensor,
        paddingMask: Tensor,
        hasPaddingMask: bool = False,
    ):
        assert tokenEmbeddings is not None, (
            f"Ensure that `tokenEmbeddings` is a `Tensor` of shape, received {type(tokenEmbeddings)}"
        )
        assert paddingMask is not None, (
            f"Ensure that `paddingMask` is a `Tensor`, received {type(paddingMask)}"
        )
        softHaltingInput = adaptiveComputationState = None
        haltMask = (1.0 - paddingMask.t().float()).contiguous()
        # TODO: In the future make sure you understand what's up with this
        # loss because, it does not accumlate it seems to simply be the
        # loss generated by the last layer. Make sure this is the
        # intended effect.
        adaptiveComputationLoss = 0.0

        layerOutput = tokenEmbeddings
        for layerIdx in range(self.numLayers):
            (
                layerOutput,
                softHaltingInput,
                adaptiveComputationState,
                adaptiveComputationLoss,
            ) = self._computeLayerOutput(
                inputTokenEmbeddings=layerOutput,
                selfAttentionInput=softHaltingInput,
                haltMask=haltMask,
                layerIdx=layerIdx,
                paddingMask=paddingMask,
                hasPaddingMask=hasPaddingMask,
                adaptiveComputationState=adaptiveComputationState,
            )

        return (
            layerOutput,
            softHaltingInput,
            adaptiveComputationLoss,
        )

    def _computeLayerOutput(
        self,
        inputTokenEmbeddings: Tensor,
        selfAttentionInput: Optional[Tensor],
        haltMask: Tensor,
        layerIdx: int,
        paddingMask: Tensor,
        hasPaddingMask: bool,
        adaptiveComputationState: Optional[Tuple] = None,
    ):
        encoderPaddingMask = paddingMask if hasPaddingMask else None
        if self.haltingFlag:
            (
                adaptiveComputationState,
                layerOutput,
                selfAttentionInput,
                adaptiveComputationLoss,
            ) = self.encoderModel.forward(
                previousAdaptiveComputationState=adaptiveComputationState,
                previousHiddenState=inputTokenEmbeddings,
                selfAttentionInput=selfAttentionInput,
                paddingMask=haltMask,
                layerIdx=layerIdx,
                encoderPaddingMask=encoderPaddingMask,
            )
        else:
            # TODO: find out of the layerIdx is needed for the encoder
            layerOutput = self.encoderModel.forward(
                inputBatch=inputTokenEmbeddings,
                selfAttentionInput=selfAttentionInput,
                haltMask=haltMask,
                layerIdx=layerIdx,
                encoderPaddingMask=encoderPaddingMask,
            )

        ffnRawOutput = None
        if isinstance(layerOutput, Tuple):
            layerOutput, ffnRawOutput = layerOutput

        self._updateDataGatheringLists(
            layerOutput=layerOutput,
            ffnRawOutput=ffnRawOutput,
            adaptiveComputationState=adaptiveComputationState,
        )

        return (
            layerOutput,
            selfAttentionInput,
            adaptiveComputationState,
            adaptiveComputationLoss,
        )

    def _updateDataGatheringLists(
        self,
        layerOutput: Tensor,
        ffnRawOutput: Optional[Tensor],
        adaptiveComputationState: Optional[Tuple] = None,
    ):
        # Not sure what `jit` stands for:  and not torch.jit.is_scripting():
        if self.returnAllHiddensFlag:
            self.encoderStatesList.append(layerOutput)
            self.ffnRawOutputList.append(ffnRawOutput)

        if self.haltingFlag and self.gatherVusalizationDataFlag:
            (
                stateIndex,
                haltingMaskLogits,
                accumulatedHiddenState,
                accumulatedExpectedDepth,
            ) = adaptiveComputationState

            haltingMaskLogits = torch.exp(haltingMaskLogits)
            self.haltingMaksLogitsList.append((1 - torch.exp(haltingMaskLogits))[:, 0])
            self.haltedMasksList.append(
                haltingMaskLogits < (1 - self.encoderModel.haltingThreshold)
            )
            self.routeIndexesList.append(
                self.encoderModel.model.ffnModel.inputExperts.numExperts
            )

    def _prepareOutput(
        self,
        layersOutput: Tensor,
        sourceTokens: Tensor,
        softHaltingInput: Optional[Tensor],
        adaptiveComputationTimeLoss: Optional[Tensor],
    ):
        totalEncoderLoss = 0.0
        if self.haltingFlag:
            layersOutput = softHaltingInput
            totalEncoderLoss += (
                self.dynamicHaltingLossWeight * adaptiveComputationTimeLoss
            )

        if self.normalizeBeforeFlag:
            layersOutput = self.transformerLayerNorm(layersOutput)

        sourceLengths = None
        if sourceTokens is not None:
            sourceLengths = (
                sourceTokens.ne(self.paddingIndex)
                .sum(dim=1, dtype=torch.int32)
                .reshape(-1, 1)
                .contiguous()
            )

        return layersOutput, totalEncoderLoss, sourceLengths

    def _storeVusalisationData(self) -> None:
        if self.haltingFlag and self.gatherVusalizationDataFlag:
            self.visualiseHaltedVisualisationTensor = torch.stack(
                self.haltingMaksLogitsList
            )
            self.haltingMaskVisualizationTensor = torch.stack(self.haltedMasksList)
            self.topKTensorVisualizationTensor = torch.stack(self.routeIndexesList)


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


class LayerDropModuleList(nn.ModuleList):
    def __init__(self, probability, modules=None):
        super().__init__(modules)
        self.probability = probability

    def __iter__(self):
        dropoutProbaiblity = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropoutProbaiblity[i] > self.probability):
                yield m
