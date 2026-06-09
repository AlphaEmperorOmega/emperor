# Emperor

Emperor is a neural module composition system where reusable blocks can be wrapped, stacked, routed, and adaptively controlled.

## Language

**Layer Controller**:
An optional gate or halting mechanism owned by a single layer's normal forward pipeline.
_Avoid_: Recurrent controller, loop controller

**Recurrent Controller**:
An optional gate or halting mechanism owned by a recurrent wrapper that repeatedly applies a reusable block.
_Avoid_: Layer controller

**Recurrent Gate**:
An optional recurrent controller that interpolates between the previous hidden state and the recurrent step candidate.
_Avoid_: Output multiplier

**Recurrent Layer**:
A wrapper that applies one reusable block repeatedly while keeping its recurrent control separate from the wrapped block's own controls.
_Avoid_: Universal layer, loop stack

**Recurrent Step**:
One application of a recurrent layer's reusable layer block.
_Avoid_: Stack layer, copied layer

**Maximum Recurrent Steps**:
The hard upper bound on how many recurrent steps a recurrent layer may perform.
_Avoid_: Unbounded recurrence

**Stable Recurrent Hidden Dimension**:
A recurrent layer's hidden feature dimension remains unchanged across recurrent steps.
_Avoid_: Step-wise dimension change

**Feature-Last Hidden State**:
A hidden tensor whose final dimension is the model feature dimension.
_Avoid_: Fixed-rank hidden state

**Maximum Grid Capacity**:
The x/y/z coordinate bounds a neuron cluster may ever instantiate.
_Avoid_: Initial cluster size when referring to the maximum

**Initialized Neuron Set**:
The sparse set of neuron coordinates built when a cluster is constructed.
_Avoid_: Full grid when only part of the maximum capacity is instantiated

**Entry Router**:
The sampler that chooses which initialized z=1 entry-plane neurons first process each feature vector.
_Avoid_: Fixed start neuron

**Escape Route**:
A selected route whose coordinate is missing or outside maximum grid capacity; it contributes the unchanged source hidden state and can end that item's route.
_Avoid_: Dropped branch, invalid zero output

**Cluster Route Halting**:
The route-level halting controller owned by a neuron cluster, independent from layer and recurrent-layer halting state.
_Avoid_: Layer halting, shared halting

**Auxiliary Loss Accumulation**:
Additional training costs from composed controls are accumulated alongside the hidden state.
_Avoid_: Loss replacement

**Itemwise Halting**:
Halting that tracks completion independently for each non-feature position while recurrent steps remain vectorized over the full tensor.
_Avoid_: Token loop, global batch halt

**Halted Hidden State**:
A hidden state position that has stopped changing because itemwise halting has completed for that position.
_Avoid_: Inactive token, skipped token

**Controller State Separation**:
Layer-level controller state and recurrent-level controller state are kept independent even when both exist.
_Avoid_: Shared halting state

**Reusable Layer Block**:
A layer or layer stack that can contain any supported model family behind the layer abstraction.
_Avoid_: Arbitrary module, raw model

**Block Config**:
The configuration for the reusable layer block wrapped by a recurrent layer.
_Avoid_: layer config when referring to the wrapper field

**Boundary Projection**:
A model boundary transform that changes feature width when data enters or leaves a hidden representation.
_Avoid_: Boundary model, input/output model when referring to projection

**Input Boundary Projection**:
The boundary projection that converts external input features into hidden features.
_Avoid_: Input model

**Output Boundary Projection**:
The boundary projection that converts hidden features into task output features.
_Avoid_: Output model

**Adaptive Boundary Projection**:
A boundary projection whose parameters may be adaptively controlled independently of the hidden representation.
_Avoid_: Reusing hidden adaptive settings implicitly

**Model Visualizer**:
A browser-based inspection tool for exploring experiment presets, config overrides, and the resulting model structure.
_Avoid_: Training dashboard, TensorBoard replacement

**Viewer**:
The repository area that owns the Model Visualizer product without becoming part of the Emperor algorithm library.
_Avoid_: Model package, algorithm component

**Training Run**:
A single Viewer-planned or Viewer-executed training unit for one preset, one dataset, and one generated config/search combination.
_Avoid_: Experiment when referring to a table row in Viewer training progress

**Training Job**:
A Viewer-started training process that executes a Training Run Plan.
_Avoid_: Experiment when referring to one Start Training action

**Training Command**:
A Viewer-displayed shell command that describes how to launch a target or Training Run from the command line.
_Avoid_: Script when referring to the displayed command text

**Config Snapshot**:
A user-created Viewer training configuration derived from one preset and a set of config overrides.
_Avoid_: Custom config, dynamic config, training config stack

**Training Run Plan**:
The Viewer's pre-flight and live list of Training Runs that will be executed by a Training Job.
_Avoid_: Metrics when referring to planned run visibility

**Training Run Progress**:
The Viewer state describing a Training Run's lifecycle status and epoch completion.
_Avoid_: Metrics when referring to run lifecycle state

**Random Training Run Plan**:
A Training Run Plan whose search combinations are sampled before a Training Job starts.
_Avoid_: Unknown random runs

## Relationships

- A **Recurrent Layer** wraps exactly one **Reusable Layer Block**.
- A **Block Config** identifies the **Reusable Layer Block** for a **Recurrent Layer**.
- A **Recurrent Layer** is a peer of a layer and a layer stack in the base layer language.
- A **Recurrent Layer** may own one **Recurrent Controller**.
- A wrapped block may own its own **Layer Controller** independently of the **Recurrent Controller**.
- A **Recurrent Layer** reuses the same **Reusable Layer Block** across all **Recurrent Steps**.
- A **Recurrent Layer** always has **Maximum Recurrent Steps**.
- A **Recurrent Layer** has a **Stable Recurrent Hidden Dimension**.
- A **Recurrent Layer** owns the stable dimension passed to its reusable layer block.
- A **Recurrent Layer** consumes a **Feature-Last Hidden State**.
- A **Recurrent Controller** uses **Itemwise Halting** when halting is enabled.
- A **Recurrent Controller** uses the same halting language as existing layer controllers.
- A **Recurrent Gate** applies only when a recurrent gate configuration is provided.
- A **Halted Hidden State** is preserved across later **Recurrent Steps**.
- A **Recurrent Controller** contributes to **Auxiliary Loss Accumulation** without replacing existing auxiliary loss.
- A **Recurrent Layer** preserves **Controller State Separation** from the wrapped block.
- An **Input Boundary Projection** and an **Output Boundary Projection** frame a classifier's hidden representation.
- An **Adaptive Boundary Projection** owns adaptive behavior separately from the hidden representation.
- A boundary projection's adaptive behavior is valid only when that boundary projection is adaptive.
- A **Model Visualizer** inspects experiment presets and config overrides without training a model.
- A **Viewer** may depend on **Emperor** and model packages, but **Emperor** does not depend on the **Viewer**.
- The **Viewer** dependency direction is enforced by backend architecture tests; core packages expose contracts instead of importing Viewer APIs.
- A **Viewer** **Training Run** belongs to one selected preset, one selected dataset, and one generated config/search combination.
- A **Training Command** may describe either the selected Viewer target or one **Training Run**.
- A **Config Snapshot** belongs to exactly one selected preset.
- A **Config Snapshot** is scoped to one Viewer session.
- A **Config Snapshot** expands across selected datasets when it contributes to a **Training Run Plan**.
- A **Config Snapshot** may contribute Training Runs to a **Training Run Plan**.
- A **Training Run Plan** contains every **Training Run** planned for one **Training Job**.
- A **Training Job** executes its **Training Run Plan** in the displayed order.
- The **Viewer** presents at most one active **Training Job** in the Training panel.
- **Training Run Progress** belongs to exactly one **Training Run**.
- A **Random Training Run Plan** may be resampled before its **Training Job** starts.
- A neuron cluster's `x/y/z_axis_total_neurons` describe **Maximum Grid Capacity**.
- A neuron cluster's `initial_*_axis_total_neurons` describe the starting **Initialized Neuron Set**.
- An **Entry Router** routes each feature vector into initialized entry-plane neurons before recurrent traversal.
- An **Escape Route** keeps its probability mass and contributes the unchanged source hidden state.
- **Cluster Route Halting** accumulates route-level candidates without sharing state with layer or recurrent controllers.

## Example dialogue

> **Dev:** "If a wrapped block already has a gate, does the recurrent wrapper reuse that gate?"
> **Domain expert:** "No — the wrapped block's gate is a **Layer Controller**; the wrapper's gate is a **Recurrent Controller**."

## Flagged ambiguities

- "gate and halting" can mean either layer-level control or recurrent-loop control — resolved: these are independent concepts.
- "input model" and "output model" were used for classifier boundary transforms — resolved: use **Input Boundary Projection** and **Output Boundary Projection**.
- "custom config", "dynamic config", and "training config stack" were used for user-created Viewer training variants — resolved: use **Config Snapshot** for the saved variant and **Training Run Plan** for the executable list.
