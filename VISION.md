# Vision

Emperor has two goals that build on each other.

## Goal 1 — A test-time training model built from interconnected neurons

The target architecture is a cluster of small models ("neurons") arranged in a 3D spatial grid. Each neuron connects to its neighbors via learned routers that decide where signals travel next.

### Routing

When a neuron has few neighbors it uses a single router; when it has many it uses two — one that selects a direction and one that selects a neuron within that direction. The final routing probability is the product of both, and it scales the neuron's output just like in Mixture of Experts.

### Processing

The core of each neuron (the "nucleus") is a sparse universal transformer that uses [soft mixture of experts](https://arxiv.org/pdf/2308.00951) instead of standard MoE. This gives each neuron a compact but expressive processing unit.

### Persistent hidden state

Every neuron maintains a hidden state that persists across batches — unlike a standard RNN, it never resets. Each time a neuron processes a batch, the hidden state is updated with new information while old information decays through a bounded nonlinearity (e.g. tanh), allowing the neuron to accumulate experience over training.

### Two levels of adaptive computation

1. **Across neurons (inter-neuron).** A signal can be routed through as many neurons as the network decides. A halting mechanism determines when processing is complete, and the final representation — regardless of which neuron produced it — is passed to the cluster's output layer for prediction (classification, next-token, etc.).

2. **Within each neuron (intra-neuron).** The nucleus runs as a universal transformer with up to 10 recurrent steps. Adaptive computation can halt early if a single pass is sufficient. Each step can maintain its own persistent hidden state, allowing different steps to store different types of experience across batches.

### Residual connections

Residual connections between neurons are supported, with [attention residuals](https://arxiv.org/pdf/2603.15031) planned as an additional option alongside standard additive residuals.

### Input-dependent parameters

Input-dependent parameters can be injected throughout the architecture to allow per-sample specialization of weights and biases.

## Goal 2 — A hierarchical configuration framework for rapid prototyping

Emperor is also a modular framework where every architectural choice — attention type, feed-forward variant, routing strategy, adaptive computation, normalization, residual connections — is a swappable configuration option. A hierarchical preset system (see `models/linear/presets.py`) lets you compose end-to-end models by mixing and matching these building blocks.

This means you can quickly spin up a transformer with, say, soft MoE feed-forward layers, mixture-of-attention-heads, stick-breaking halting, and input-dependent bias generation — all by changing configuration values rather than writing new model code.

Goal 2 is the framework that enables Goal 1: the neuron cluster architecture is assembled from the same configurable components that power every other reference model in the project.

## Training lifecycle

The model trains in two phases:

### Phase 1 — Initial training

The cluster is trained normally on a dataset. All neurons receive gradients and update their weights through standard backpropagation.

### Phase 2 — Test-time training

Once deployed, the model continues to learn from incoming data. This is what makes it a test-time training model — it does not stop learning after initial training.

**Structural growth.** New neurons are added at the edges of the grid. Because of adaptive computation, most signals are processed by a core set of neurons and exit early to make a prediction, while some signals propagate outward to the new edge neurons. Edge neurons are trained directly on incoming data through normal gradient updates. The exact mechanism for directing training signal to edge neurons is an open design question.

**Neuron freezing.** As neurons mature, their weights are frozen — they no longer receive standard gradient updates. However, frozen does not mean static.

**Sparse training.** Every neuron — including frozen ones — has a memory module that accumulates information from signals passing through it. After a fixed number of signals (e.g. 100), the accumulated knowledge is flushed and used to perform a single sparse weight update on that neuron only. This allows frozen neurons to slowly adapt to distributional shifts without continuous gradient flow. Each neuron manages its own accumulation cycle independently.

In summary: edge neurons learn fast through direct training, core neurons learn slowly through periodic sparse updates, and the cluster grows structurally to absorb new knowledge over time.

## Target experiment

Once the neuron cluster architecture is complete, the first planned experiment is to pit two of these models against each other in a game of chess — to observe how structural growth and persistent hidden state affect learned strategy over time.
