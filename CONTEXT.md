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

**Gate Config**:
A shared controller configuration wrapper that groups a gate's behavior option with the layer-stack model config that produces gate values.
_Avoid_: Separate model config and gate option fields

**Layer Gate Option**:
The configured behavior for a layer-owned gate output, such as raw multiplication, squashed multiplication, or residual blending.
_Avoid_: Recurrent gate option

**Recurrent Gate Option**:
The configured behavior for a recurrent gate output, such as sigmoid or normalized tanh interpolation between candidate and previous hidden state.
_Avoid_: Layer gate option

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

**Dynamic Memory**:
An optional layer-owned module that generates a memory representation from the current hidden state and merges it back into that state.
_Avoid_: External cache when referring to the layer module

**Memory Position**:
The configured point where Dynamic Memory is applied in a layer, either before or after the affine model.
_Avoid_: Memory type

**Memory Generator**:
The internal reusable layer stack that produces memory, gate, weight, query, key, value, decoder, or output projections for Dynamic Memory.
_Avoid_: Main model

**Test-Time-Training Memory**:
Dynamic Memory mode that adapts the memory generator parameters with an inner reconstruction objective during forward execution.
_Avoid_: Normal training step

**Shared Memory**:
One Dynamic Memory module reused by every layer in a layer stack.
_Avoid_: Per-layer memory

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

**Training Setup Sidebar**:
The Viewer training area for choosing what will contribute to a Training Run Plan.
_Avoid_: Main menu, footer menu

**Training Run List**:
The Viewer training area that presents Training Runs from a Training Run Plan as rows.
_Avoid_: Runs window

**Expanded Run View**:
A larger Viewer training surface for inspecting the Training Run List without changing Training Run Plan inputs.
_Avoid_: Training Progress popup when referring to setup controls

**Training Status Sidebar**:
The Viewer training area that summarizes Training Job state, Training Run Plan state, and training logs.
_Avoid_: Right footer sidebar

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
- **Dynamic Memory** consumes and returns a **Feature-Last Hidden State** with the same final dimension.
- A **Memory Position** determines whether **Dynamic Memory** sees the layer input dimension or output dimension.
- A **Memory Generator** is built from a supported layer or layer-stack generator config.
- **Test-Time-Training Memory** requires both an inner-loop learning rate and inner-step count, and both must be positive.
- **Shared Memory** is valid only when every layer in the stack has a stable memory dimension.
- A **Recurrent Controller** uses **Itemwise Halting** when halting is enabled.
- A **Recurrent Controller** uses the same halting language as existing layer controllers.
- A **Recurrent Gate** applies only when a recurrent gate configuration is provided.
- A **Gate Config** is shared by layer gates, recurrent gates, and shared layer-stack gates; its owner determines which gate option enum is valid.
- A **Layer Gate Option** configures only a **Layer Controller** gate.
- A **Recurrent Gate Option** configures only a **Recurrent Controller** gate.
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
- A **Config Snapshot** can be drafted from a selected preset without inheriting the current override draft.
- A **Config Snapshot** can be edited in place or duplicated into a separate draft.
- A **Config Snapshot** is scoped to one Viewer session.
- A **Config Snapshot** expands across selected datasets when it contributes to a **Training Run Plan**.
- A **Config Snapshot** may contribute Training Runs to a **Training Run Plan**.
- A **Training Run Plan** contains every **Training Run** planned for one **Training Job**.
- A **Training Job** executes its **Training Run Plan** in the displayed order.
- A **Training Job** does not change its **Training Run Plan** after it starts.
- The **Viewer** presents at most one active **Training Job** in the Training panel.
- **Training Run Progress** belongs to exactly one **Training Run**.
- The **Training Setup Sidebar** selects inputs for a **Training Run Plan**.
- The **Training Run List** displays the **Training Runs** in a **Training Run Plan**.
- The **Expanded Run View** presents the **Training Run List** in a larger surface.
- The **Training Status Sidebar** summarizes the current **Training Job** and **Training Run Plan**.
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
- "gate option" can mean either **Layer Gate Option** or **Recurrent Gate Option** — resolved: name the controller owner.
- "input model" and "output model" were used for classifier boundary transforms — resolved: use **Input Boundary Projection** and **Output Boundary Projection**.
- "custom config", "dynamic config", and "training config stack" were used for user-created Viewer training variants — resolved: use **Config Snapshot** for the saved variant and **Training Run Plan** for the executable list.
- "create snapshot from preset" can mean either inheriting the current override draft or starting from preset defaults — resolved: a **Config Snapshot** drafted from a preset starts from that preset's defaults.
