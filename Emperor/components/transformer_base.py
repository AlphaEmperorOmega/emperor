import torch
import torch.nn as nn
import torch.nn.functional as F

from Emperor.base.utils import Module
from typing import TYPE_CHECKING, _MISSING_TYPE, Dict, Optional, List, Tuple, Any

from Emperor.components.transformer_decoder import TransformerDecoderBase
from Emperor.components.transformer_encoder import TransformerEncoderBase

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


def get_moe_modules(model: nn.Module) -> List[MoE]:
    moes = set()
    moe_dict = {}
    modules = model.named_modules()
    for n, m in modules:
        if isinstance(m, MoE) and m not in moes:
            moes.add(m)
            moe_dict[n] = m
    return moe_dict


class TransformerModelBase(Module):
    def __init__(self, cfg: "ModelConfig", encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        # self.supports_align_args = True
        self.supportsAlignArguments = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def buildModel(cls, cfg: "ModelConfig", task):
        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        sourceDict, targetDict = task.source_dictionary, task.target_dictionary

        if cfg.shareAllEmbeddings:
            if sourceDict != targetDict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embeddingDim != cfg.decoder.embeddingDim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embeddingPath and (
                cfg.decoder.embeddingPath != cfg.encoder.embeddingPath
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoderEmbeddingModel = cls.buildEmbeddingModel(
                cfg, sourceDict, cfg.encoder.embeddingDim, cfg.encoder.embeddingDim
            )
            decoderEmbeddingModel = encoderEmbeddingModel
            cfg.shareEncoderInputOuputEmbeddings = True
        else:
            encoderEmbeddingModel = cls.buildEmbeddingModel(
                cfg, sourceDict, cfg.encoder.embeddingDim, cfg.encoder.embeddingDim
            )
            decoderEmbeddingModel = cls.buildEmbeddingModel(
                cfg, targetDict, cfg.decoder.embeddingDim, cfg.decoder.embeddingDim
            )
        if cfg.offloadActivations:
            cfg.checkpointActivations = True  # offloading implies checkpointing

        encoderModel = cls.buildEncoder(cfg, sourceDict, encoderEmbeddingModel)
        decoderModel = cls.buildDecoder(cfg, targetDict, decoderEmbeddingModel)
        model = cls(cfg, encoderModel, decoderModel)
        # cls.ensureAuxiliaryLossExistsInMoe(model)
        return model

    def ensureAuxiliaryLossExistsInMoe(model: TransformerModelBase) -> None:
        model.moeModules = get_moe_modules(model)
        for n in model.moeModules:
            assert model.moeModules[n].accumulativeAuxiliaryLoss

    @classmethod
    def buildEmbeddingModel(
        cls,
        cfg: "ModelConfig",
        dictionary,
        embeddingDim,
        path=None,
    ):
        numEmbeddings = len(dictionary)
        paddingIdx = dictionary.pad()

        embeddingModel = Embedding(numEmbeddings, embeddingDim, paddingIdx)
        if path:
            embed_dict = parse_embedding(path)
            load_embedding(embed_dict, dictionary, embeddingModel)
        return embeddingModel

    @classmethod
    def buildEncoder(cls, cfg, sourceDict, embeddingModel):
        return TransformerEncoderBase(cfg, sourceDict, embeddingModel)

    @classmethod
    def buildDecoder(cls, cfg, targetDict, embeddingModel):
        return TransformerDecoderBase(
            cfg,
            targetDict,
            embeddingModel,
            useEncoderAttentionFlag=cfg.useEncoderAttentionFlag,
        )

    def forward(
        self,
        sourceTokens,
        sourceLengths,
        targetTokens,
        featuresOnlyFlag: bool = False,
        alignmentLayer: Optional[int] = None,
        alignmentHeads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoderOutput = self.encoder(
            sourceTokens=sourceTokens,
            sourceSequenceLengths=sourceLengths,
        )
        decoderOutput = self.decoder(
            targetTokens=targetTokens,
            encoderOutputDict=encoderOutput,
            featuresOnlyFlag=featuresOnlyFlag,
            alignmentLayer=alignmentLayer,
            alignmentHeads=alignmentHeads,
        )
        return decoderOutput

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(numEmbeddings, embeddingDim, paddingIdx):
    m = nn.Embedding(numEmbeddings, embeddingDim, padding_idx=paddingIdx)
    nn.init.normal_(m.weight, mean=0, std=embeddingDim**-0.5)
    nn.init.constant_(m.weight[paddingIdx], 0)
    return m


class FairseqDataclass:
    """fairseq base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")

    @classmethod
    def from_namespace(cls, args):
        if isinstance(args, cls):
            return args
        else:
            config = cls()
            for k in config.__dataclass_fields__.keys():
                if k.startswith("_"):
                    # private member, skip
                    continue
                if hasattr(args, k):
                    setattr(config, k, getattr(args, k))

            return config


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding
