import torch
from torch import Tensor
from torch.nn import Sequential, Parameter, Conv2d, Flatten, Dropout
from Emperor.base.utils import Module


class PatchEmbeddingConv(Module):
    def __init__(
        self,
        inputChannels: int,
        embeddingDim: int,
        patchSize: int,
        numPatches: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patchModel = Sequential(
            Conv2d(
                in_channels=inputChannels,
                out_channels=embeddingDim,
                # if kernel_size = stride -> no overlap
                kernel_size=patchSize,
                stride=patchSize,
            ),
            Flatten(2),
        )
        classTokenInit = torch.randn((1, 1, embeddingDim))
        self.classToken = Parameter(classTokenInit, requires_grad=True)
        positionEmbeddingsInit = torch.randn((1, numPatches + 1, embeddingDim))
        self.positionEmbeddings = Parameter(positionEmbeddingsInit, requires_grad=True)
        self.dropoutModule = Dropout(dropout)

    def forward(self, inputBatch: Tensor):
        inputBatch = self.patchModel(inputBatch)
        inputBatch = inputBatch.permute(0, 2, 1)
        classToken = self.classToken.expand(inputBatch.size(0), -1, -1)
        inputBatch = torch.cat([classToken, inputBatch], dim=1)
        inputBatch = self.positionEmbeddings + inputBatch
        inputBatch = self.dropoutModule(inputBatch)
        return inputBatch
