import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, export
from torch.nn.modules import transformer
from Emperor.base.utils import Module
from Emperor.components.sut_layer import TransformerEncoderLayerBase
from Emperor.config import ModelConfig

from typing import TYPE_CHECKING, Dict, Optional, List, Tuple

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class TransformerEncoderBase(Module):
    def __init__(self, cfg: "ModelConfig", tokenEmbeddingModule):
        super().__init__()
        self.register_buffer("version", torch.Tensor([1]))

        self.cfg = cfg
        self.maxSourceEmbeddingPositions = cfg.maxSourceEmbeddingPositions
        self.numLayers = cfg.numLayers
        self.dynamicHaltingLossWeight = cfg.dynamicHaltingLossWeight
        self.embeddingDropoutProbability = 0.0
        self.gatherVusalisationDataFlag = False
        self.embeddingLayerNormFlag = False
        self.positionalEmbeddingFlag = True
        self.normalizeBeforeFlag = True
        self.quantNoiseFlag = False

        self.tokenEmbeddingModule = tokenEmbeddingModule
        self.paddingIndex = self.embeddingModule.padding_idx
        self.embeddingDim = self.embeddingModule.embedding_dim
        self.embeddingTokensWeight = (
            1.0 if cfg.scaleEmbeddingTokensFlag else math.sqrt(self.embeddingDim)
        )

        self._resetDataGathering()
        self._initTransformerModules()
        self._initTransformerLayer()
        self._initEmbeddingModules()

    def _resetDataGathering(self):
        self.encoderStatesList = []
        self.ffnRawOutputList = []
        self.acc_psList = []
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

        self.halting = False
        if self.cfg.encoder.halting:
            self.halting = True
            self.encoderModel = ACTWrapper(
                self.encoderModel, halting_dropout=cfg.halting_dropout
            )

    def _createTransformerModelHook(self):
        transformerLayer = TransformerEncoderLayerBase(
            self.cfg, returnRawFFNOutputFlag=self.return_fc
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
        self.embedingDropoutModule = nn.Dropout(self.embeddingDropoutProbability)

        if self.embeddingLayerNormFlag:
            self.embeddingLayerNorm = nn.LayerNorm(self.ebme)

        if self.positionalEmbeddingFlag:
            self.positionalEmbedding = PositionalEmbedding(
                self.maxSourceEmbeddingPositions,
                self.embeddingDim,
                self.paddingIndex,
                learned=cfg.encoder.learned_pos,
            )

        self.quantNoise = None
        # TODO: figure out later how this works
        # if not cfg.adaptive_input and cfg.quantNoise.pq > 0:
        #     self.quantNoise = apply_quant_noise_(
        #         nn.Linear(embed_dim, embed_dim, bias=False),
        #         cfg.quantNoise.pq,
        #         cfg.quantNoise.pq_block_size,
        #     )

    def forward(
        self,
        sourceTokens: Tensor,
        sourceSequenceLengths: Optional[torch.Tensor] = None,
        returnAllHiddens: bool = False,
        tokenEmbeddings: Optional[torch.Tensor] = None,
    ):
        self._resetDataGathering()
        x, paddingMask, rawEncoderEmbedding, hasPaddingMask = self._computeInputTokens(
            sourceTokens=sourceTokens,
            tokenEmbeddings=tokenEmbeddings,
            returnAllHiddens=returnAllHiddens,
        )

        inputEmbeddings, ffnRawOutput, act_state, act_loss = (
            self._computeAllEncoderLayersOutput(
                inputEmbeddings=x,
                paddingMask=paddingMask,
                hasPaddingMask=hasPaddingMask,
            )
        )

        self._computeModelOutput(
            x=x,
            sourceTokens=sourceTokens,
            softHaltingInput=soft_halt_x,
            act_loss=act_loss,
        )

        self._updateVusalisationData()

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoderPaddingMask],  # B x T
            "encoder_embedding": [rawEncoderEmbedding],  # B x T x C
            "encoder_states": self.encoderStatesList,  # List[T x B x C]
            "fc_results": ffnRawOutput,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "encoder_loss": [totalEncoderLoss],
            "encoder_expected_halt": act_loss,
        }

    def _computeModelOutput(
        self,
        x,
        sourceTokens,
        softHaltingInput,
        act_loss,
    ):
        totalEncoderLoss = 0
        if self.halting:
            x = softHaltingInput
            totalEncoderLoss += self.dynamicHaltingLossWeight * act_loss

        if self.normalizeBeforeFlag is not None:
            x = self.transformerLayerNorm(x)

        sourceLengths = (
            sourceTokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )

        return x, totalEncoderLoss, sourceLengths

    def _computeInputTokens(
        self,
        sourceTokens: Tensor,
        tokenEmbeddings: Optional[torch.Tensor] = None,
        returnAllHiddens: bool = False,
    ):
        sequenceLengthPaddingMask = sourceTokens.eq(self.paddingIndex)
        hasPaddingMask = (
            sourceTokens.device.type == "xla" or sequenceLengthPaddingMask.any()
        )

        x, rawEncoderEmbedding, _ = self._computeTokenEmbedding(
            sourceTokens, tokenEmbeddings
        )

        if hasPaddingMask:
            x = x * (1 - sequenceLengthPaddingMask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if returnAllHiddens:
            self.encoderStatesList.append(x)

        return x, sequenceLengthPaddingMask, rawEncoderEmbedding, hasPaddingMask

    def _computeTokenEmbedding(
        self,
        sourceTokens: Tensor,
        tokenEmbeddings: Optional[torch.Tensor] = None,
    ):
        if tokenEmbeddings is None:
            tokenEmbeddings = self.tokenEmbeddingModule(sourceTokens)
        x = rawInputTokens = self.embeddingTokensWeight * tokenEmbeddings

        positionalEmbedding = None
        if self.positionalEmbeddingFlag:
            positionalEmbedding = self.positionalEmbedding(sourceTokens)
            x += positionalEmbedding

        if self.embeddingLayerNormFlag:
            x = self.embeddingLayerNorm(x)

        x = self.embedingDropoutModule(x)

        if self.quantNoise is not None:
            x = self.quantNoise(x)

        return x, rawInputTokens, positionalEmbedding

    def _computeAllEncoderLayersOutput(
        self,
        inputEmbeddings,
        paddingMask,
        hasPaddingMask: bool = False,
    ):
        softHaltingInput = ffnRawOutput = act_state = None

        haltMask = (1.0 - paddingMask.t().float()).contiguous()

        for layerIdx in range(self.numLayers):
            inputEmbeddings, ffnRawOutput, act_state, act_loss = (
                self._computeLayerOutput(
                    inputEmbeddings=inputEmbeddings,
                    selfAttentionInput=softHaltingInput,
                    haltMask=haltMask,
                    layerIdx=layerIdx,
                    paddingMask=paddingMask,
                    hasPaddingMask=hasPaddingMask,
                    act_state=act_state,
                )
            )

        return inputEmbeddings, ffnRawOutput, act_state, act_loss

    def _computeLayerOutput(
        self,
        inputEmbeddings: Tensor,
        selfAttentionInput: Optional[Tensor],
        haltMask: Tensor,
        layerIdx: int,
        paddingMask: Optional[Tensor],
        hasPaddingMask: bool,
        returnAllHiddens: bool = False,
        act_state: Optional[Tuple] = None,
    ):
        act_loss = 0.0
        encoderPaddingMask = paddingMask if hasPaddingMask else None
        if self.halting:
            # TODO: Inprove the following after you work on "ACTWrapper"
            act_state, layerOutput, soft_halt_x, act_loss = self.encoderModel.forward(
                act_state,
                inputTokens,
                self_attn_input=soft_halt_x,
                pad_mask=haltMask,
                layer_idx=layerIdx,
                encoder_padding_mask=encoderPaddingMask,
            )
        else:
            # TODO: find out of the layerIdx is needed for the encoder
            layerOutput = self.encoderModel.forward(
                inputBatch=inputEmbeddings,
                selfAttentionInput=selfAttentionInput,
                haltMask=haltMask,
                # layerIdx=layerIdx,
                encoderPaddingMask=encoderPaddingMask,
            )

        ffnRawOutput = None
        if isinstance(layerOutput, tuple):
            layerOutput, ffnRawOutput = layerOutput

        self._updateDataGathering(
            inputEmbeddings=inputEmbeddings,
            ffnRawOutput=ffnRawOutput,
            act_state=act_state,
            returnAllHiddens=returnAllHiddens,
        )

        return layerOutput, ffnRawOutput, act_state, act_loss

    def _updateDataGathering(
        self,
        inputEmbeddings: Tensor,
        ffnRawOutput: Optional[Tensor],
        act_state: Optional[Tuple] = None,
        returnAllHiddens: bool = False,
    ):
        if returnAllHiddens and not torch.jit.is_scripting():
            self.encoderStatesList.append(inputEmbeddings)
            self.ffnRawOutputList.append(ffnRawOutput)

        if self.halting and self.gatherVusalisationDataFlag:
            (i, log_never_halt, acc_h, acc_expect_depth) = act_state
            p_never_halt = torch.exp(log_never_halt)
            self.acc_psList.append((1 - torch.exp(log_never_halt))[:, 0])
            self.haltedMasksList.append(
                p_never_halt < (1 - self.halting_universal_layer.threshold)
            )
            self.routeIndexesList.append(self.universal_layer.moe.top_k_indices)

    def _updateVusalisationData(self) -> None:
        if self.halting and self.gatherVusalisationDataFlag:
            self.visualiseHaltedVisualisationTensor = torch.stack(self.acc_psList)
            self.haltingMaskVisualizationTensor = torch.stack(self.haltedMasksList)
            self.topKTensorVisualizationTensor = torch.stack(self.routeIndexesList)


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, numEmbeddings: int, embedingDim: int, paddingIdx: int):
        super().__init__(numEmbeddings, embedingDim, paddingIdx)
        self.onnxTrace = False
        if self.padding_idx:
            self.maxPositions = self.num_embeddings - self.padding_idx
        else:
            self.maxPositions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        assert (positions is None) or (self.padding_idx is None), (
            "If positions is pre-computed then padding_idx should not be set."
        )

        if positions is None:
            if incrementalState is not None:
                fillValue = int(self.padding_idx + input.size(1))
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(fillValue)
            else:
                mask = torch.ne(padding_idx).int()
                positions = (
                    torch.cumsum(mask, dim=1).type_as(mask) * mask
                ).long() + padding_idx

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
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
