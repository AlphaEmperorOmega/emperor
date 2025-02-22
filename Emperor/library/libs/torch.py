import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

from typing import Optional, List, Tuple


class Library:
    Tensor = torch.Tensor
    Parameter = nn.Parameter
    Long = torch.long
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def areTensorsIdentical(firstTensor, secondTensor) -> bool:
        return torch.eq(firstTensor, secondTensor).all()

    @staticmethod
    def tensorProduct(firstTensor, secondTensor):
        return torch.matmul(firstTensor, secondTensor)

    @staticmethod
    def matrixProduct(firstTensor, secondTensor):
        return torch.mm(firstTensor, secondTensor)

    @staticmethod
    def vectorOuterProduct(firstTensor, secondTensor):
        return torch.outer(firstTensor, secondTensor)

    @staticmethod
    def softmax(inputDim, dim=-1):
        return torch.softmax(inputDim, dim=dim)

    @staticmethod
    def getTopProbabilityAndIndex(probabilities, dim=-1):
        return torch.max(probabilities, dim=dim)

    @staticmethod
    def topk(inputTensor, topk, dim=-1):
        return inputTensor.topk(topk, dim=dim)

    @staticmethod
    def ones(shape: Tuple):
        return torch.ones(shape)

    @staticmethod
    def zeros(
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad=True,
    ):
        return torch.zeros(
            *shape, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @staticmethod
    def arange(rangeSize):
        return torch.arange(rangeSize)

    @staticmethod
    def reshape(tensor, shape: List):
        return tensor.reshape(*shape)

    @staticmethod
    def toTensor(numberList):
        return torch.tensor(numberList)  # [1,2,3,...,numberList]

    @staticmethod
    def shape(inputTensor):
        return inputTensor.shape

    @staticmethod
    def expand(inputTensor, shape: List):
        return inputTensor.expand(*shape)

    @staticmethod
    def sum(inputTensor, dim=-1, keepdim=False):
        if dim is None:
            return inputTensor.sum()
        return torch.sum(inputTensor, dim=dim, keepdim=keepdim)

    @staticmethod
    def toList(inputTensor):
        return torch.tensor(inputTensor).tolist()

    @staticmethod
    def shapeList(inputTensor):
        return Library.toList(Library.shape(inputTensor))

    @staticmethod
    def createTensor(
        vectorRange: int,
        shape: Optional[List] = None,
    ):
        vector = Library.arange(vectorRange)
        if not shape:
            return vector
        return Library.reshape(vector, shape)

    @staticmethod
    def unsqueeze(inputTensor, dim=0):
        return inputTensor.unsqueeze(dim=dim)

    @staticmethod
    def mean(inputTensor, dim=0):
        return inputTensor.mean(dim=dim)

    @staticmethod
    def var(inputTensor, dim=0):
        return inputTensor.var(dim=dim)

    @staticmethod
    def parameterWeights(*shape):
        return nn.Parameter(Library.randn(*shape))

    @staticmethod
    def transpose(inputTensor, shape: List):
        return inputTensor.transpose(*shape)

    @staticmethod
    def randn(*shape):
        return torch.randn(*shape)

    @staticmethod
    def randint(min: int, max: int, shape: Tuple):
        return torch.randint(min, max, shape)

    @staticmethod
    def Linear(inputDim, outputDim, bias=True):
        return nn.Linear(inputDim, outputDim, bias)

    @staticmethod
    def Dropout(dropoutProbability):
        return nn.Dropout(dropoutProbability)

    @staticmethod
    def Sequential(*layerList):
        return nn.Sequential(*layerList)

    @staticmethod
    def ModuleList(layerList: List):
        return nn.ModuleList(layerList)

    @staticmethod
    def Flatten():
        return nn.Flatten()

    @staticmethod
    def Tanh():
        return nn.Tanh()

    @staticmethod
    def ReLU():
        return nn.ReLU()

    @staticmethod
    def sigmoid(inputTensor):
        return torch.sigmoid(inputTensor)

    @staticmethod
    def chunk(inputTensor, numChunks, dim=-1):
        return inputTensor.chunk(numChunks, dim=dim)

    @staticmethod
    def randn_like(inputTensor):
        return torch.randn_like(inputTensor)

    @staticmethod
    def multinomial(inputTensor, numSamples):
        return torch.multinomial(inputTensor, numSamples)

    @staticmethod
    def cat(tensorList: List[Tensor], dim=-1):
        return torch.cat(tensorList, dim=dim)

    @staticmethod
    def pad(tensor: Tensor, padding: Tuple):
        return F.pad(tensor, pad=padding)

    @staticmethod
    def gather(inputTensor, dim, indices):
        return torch.gather(inputTensor, dim=dim, index=indices)

    @staticmethod
    def getSmallestDtypeNumber(inputTensor):
        return torch.finfo(inputTensor.dtype).min

    @staticmethod
    def where(condition, input, other):
        return torch.where(condition, input, other)

    @staticmethod
    def normalize(inputTensor, dim=0):
        return F.normalize(inputTensor, dim)

    @staticmethod
    def einsum(equation, *inputTensors):
        return torch.einsum(equation, inputTensors)

    @staticmethod
    def initializeWeights(*weightTensors):
        for weightTensor in weightTensors:
            if isinstance(weightTensor, Library.Parameter):
                init.xavier_uniform_(weightTensor)

            if isinstance(weightTensor, nn.Linear):
                init.xavier_uniform_(weightTensor.weight)
                if weightTensor.bias is not None:
                    init.zeros_(weightTensor.bias)
