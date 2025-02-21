# Sparse Universal Transformer Encoder Components

-

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
from torch import Tensor
from typing import Dict, Optional

from Emperor.library.choice import Library as L
from Emperor.components.sut_layer import TransformerEncoderLayerBase
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.config import ModelConfig
```

### 1.2. Setup configuration

```{python}
cfg = ModelConfig(
  embeddingDim=4,
  qkvHiddenDim=6,
  sequenceLength=7,
  attentionOutputDim=8,
  headDim=2,
  numExperts=7,
  topK=2,
)
cfg.auxiliaryLosses = AuxiliaryLosses(cfg)
cfg.moeAuxiliaryLosses = AuxiliaryLosses(cfg)
print('# Current Parameter Generator: \n', cfg.parameterGeneartorType.name)
```

### 1.3. Initialize model

```{python}
print('# embeddingDim: ', cfg.embeddingDim)
print('# qkvHiddenDim: ', cfg.qkvHiddenDim)
print('# attentionOutputDim: ', cfg.attentionOutputDim)
print('# numExperts: ', cfg.numExperts)
print('# topK: ', cfg.topK)
print('# headDim: ', cfg.headDim)
```

```{python}
model = TransformerEncoderLayerBase(
  cfg=cfg,
  embeddingDim = cfg.embeddingDim,
  returnRawFFNOutputFlag = cfg.returnRawFFNOutputFlag,
  normalizeBeforeFlag = cfg.normalizeBeforeFlag,
  activationFunction = cfg.activationFunction ,
  attnDropoutProbability = cfg.attnDropoutProbability,
  ffnDropoutProbability = cfg.ffnDropoutProbability,
)
```
