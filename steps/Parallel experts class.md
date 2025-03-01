<!--toc:start-->

- [Step by Step: `ParallelExperts`](#step-by-step-parallelexperts)
  - [1. Set up configuration](#1-set-up-configuration)
  - [2. Initialize `numExperts` parameter generator layers models: `_createExperts`](#2-initialize-numexperts-parameter-generator-layers-models-createexperts)
  - [3. Inputs](#3-inputs)
    - [3.1 Experts frequency vector that contains token count for each expert](#31-experts-frequency-vector-that-contains-token-count-for-each-expert)
      - [3.1.1 Input batch: `[batchSize * sequenceLenth, embeddingDim]`](#311-input-batch-batchsize-sequencelenth-embeddingdim)
      - [3.1.2 Expert sorted `topk` indexes: `[1, batchSize * sequenceLenth * topK]`](#312-expert-sorted-topk-indexes-1-batchsize-sequencelenth-topk)
      - [3.1.3 Expert ordered tokens for each expert: `[batchSize * sequenceLenth * topK, embeddingDim]`](#313-expert-ordered-tokens-for-each-expert-batchsize-sequencelenth-topk-embeddingdim)
    - [3.2 Experts frequency vector that contains token count for each expert](#32-experts-frequency-vector-that-contains-token-count-for-each-expert)
  - [4. Split input tensor into a list of tensors for each expert: `_splitInputExperts`](#4-split-input-tensor-into-a-list-of-tensors-for-each-expert-splitinputexperts)
    - [4.1 Convert `expertFrequency` input to list](#41-convert-expertfrequency-input-to-list)
    - [4.2 Split the expert ordered input into `numExperts` tensors](#42-split-the-expert-ordered-input-into-numexperts-tensors)
  - [5. Process expert-specific inputs: `forward`](#5-process-expert-specific-inputs-forward)
    - [5.1 Pass each tensor generated in 4.2 through it's assigned expert](#51-pass-each-tensor-generated-in-42-through-its-assigned-expert)
  - [6. Output: Concatenated list of expert outputs: `[batchSize * sequenceLenth * topK, embeddingDim]`](#6-output-concatenated-list-of-expert-outputs-batchsize-sequencelenth-topk-embeddingdim)
  - [TEST: ParallelExperts class](#test-parallelexperts-class)
  <!--toc:end-->

# Step by Step: `ParallelExperts`

This class is used to split the input into tensors assigned to each expert and process those tensors by the by it's assigned expert

```{python}
from Emperor.library.choice import Library as L
from Emperor.components.experts import ParallelExperts
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.components.layer import ParameterGeneratorLayer
from Emperor.config import ModelConfig
```

## 1. Set up configuration

```{python}
cfg = ModelConfig(
  batchSize=5,
  inputDim=8,
  topK=3,
)
cfg.auxiliaryLosses = AuxiliaryLosses(cfg)
```

## 2. Initialize `numExperts` parameter generator layers models: `_createExperts`

- experts use instances of [[Parameter generator feed forward layer]]
- experts can a parameter generator [[Parameter generators]] to generate weight and bias parameters for each input sample, the type of generators can be set in `ParallelExpertsConfig` option `parameterGeneartorType`

```{python}
experts = []
for _ in range(cfg.numExperts):
    layer = ParameterGeneratorLayer(cfg)
    experts.append(layer)

experts = L.ModuleList(experts)
print('# Expert instances: \n', experts)
```

## 3. Inputs

### 3.1 Experts frequency vector that contains token count for each expert

#### 3.1.1 Input batch: `[batchSize * sequenceLenth, embeddingDim]`

```{python}
inputBatch = L.arange(cfg.batchSize * cfg.inputDim).float()
inputBatch = inputBatch.reshape(cfg.batchSize, cfg.inputDim)
print(f"# Input batch shape: ", inputBatch.shape)
print(f"# Input batch: \n", inputBatch)
```

#### 3.1.2 Expert sorted `topk` indexes: `[1, batchSize * sequenceLenth * topK]`

- More details: [[MOE - computeGating method]]

```{python}
expertSortedTopKBatchIndexes = [3, 1, 4, 1, 2, 4, 0, 3, 0, 1, 2, 3, 4, 2, 0]
print(f"# Indexes used to assign input samples to their experts: \n", expertSortedTopKBatchIndexes)
```

#### 3.1.3 Expert ordered tokens for each expert: `[batchSize * sequenceLenth * topK, embeddingDim]`

`inputBatch` use `expertSortedTopKBatchIndexes` from [[MOE - computeGating method]] to duplicate the single input sample for the expert it was assigned to

```{python}
expertOrderedInput = inputBatch[expertSortedTopKBatchIndexes]
print(f"# Expert sorted topk input batch: ", expertOrderedInput.shape)
print(f"# Input Input assigned to it's epxerts: \n", expertOrderedInput)
```

### 3.2 Experts frequency vector that contains token count for each expert

Tensor vector that contains the number of samples assigned to a specific expert calculated in [[MOE - computeGating method]]

```{python}
expertFrequency = L.toTensor([1, 2, 3, 2, 2, 3, 2])
print(expertFrequency)
```

## 4. Split input tensor into a list of tensors for each expert: `_splitInputExperts`

### 4.1 Convert `expertFrequency` input to list

```{python}
expertFrequencyList = expertFrequency.tolist()
print(expertFrequencyList)
```

### 4.2 Split the expert ordered input into `numExperts` tensors

```{python}
expertInputs = expertOrderedInput.split(expertFrequencyList)
print(expertInputs)
```

## 5. Process expert-specific inputs: `forward`

### 5.1 Pass each tensor generated in 4.2 through it's assigned expert

```{python}
expertOutputs = []
for expertIndex in range(cfg.numExperts):
    input = expertInputs[expertIndex]
    output = experts[expertIndex](input)
    expertOutputs.append(output)
print('# Expert 1 output: \n', expertOutputs[0])
for index, expertOutput in enumerate(expertOutputs):
  print(f'# Expert {index} output shape: ', expertOutput.shape)
```

## 6. Output: Concatenated list of expert outputs: `[batchSize * sequenceLenth * topK, embeddingDim]`

```{python}
parallelExpertOutput = L.cat(expertOutputs, dim=0)
print(f'# All experts output shape: ', parallelExpertOutput.shape)
print(f'# All experts output: \n', parallelExpertOutput)
```

## TEST: ParallelExperts class

```{python}
model = ParallelExperts(cfg)
modelOutput = model(expertOrderedInput, expertFrequency)
print(f'# All experts output shape: ', modelOutput.shape)
print(f'# All experts output: \n', modelOutput)
```
