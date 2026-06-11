# Neuron Module — Feature Backlog

Candidate features for `emperor/neuron/`, collected during the 2026-06 production-readiness review.
Already implemented (not listed below): escape-driven growth placement (`escape_driven_growth_flag`),
mitosis-style initialization (`mitosis_initialization_flag`), and pruning / atrophy
(`pruning_threshold` — per-neuron atrophy counters, entry plane protected, bidirectional
state-dict reconciliation, optimizer sync removal).

---

## Structural plasticity

### Growth budget / cooldown

Only throttles today: one growth per forward, per-neuron counter reset, grid capacity.

- Config: e.g. `growth_cooldown_steps` (min training forwards between growths) and
  `max_total_growths` (global budget).
- Persistent step/growth-count buffers, early return in `__check_neuron_growth`,
  validator wiring (require `growth_threshold`, like the placement flags).
- Complexity: low — same shape as the escape/mitosis flags.

## Routing expressiveness

### Coordinate embeddings

Neurons don't know where they are; routing input carries no position. Inject an
(x, y, z) encoding into nucleus/terminal input so routing and processing are
position-aware. Complexity: medium.

### Beam continuation

Only the argmax branch continues a route; top-k branches are already processed
speculatively, then discarded. Continue 2–3 weighted branches and merge — reuses
compute already spent. Complexity: medium-high (route state becomes per-beam).

### Residual blend on processing

Hidden state is fully replaced at each hop; deep routes can wash out the input.
Optional residual mix (config flag) on `process_signal` output. Complexity: low.

### Explicit exit plane

"Escape anywhere" is currently the only way out. A designated readout plane
(e.g. z = max) gives routes a destination, with an optional readout head.
Complexity: medium.

### Cluster-level load-balance loss

The entry sampler balances entry only; nothing balances interior traffic.
Coefficient-of-variation auxiliary loss over per-neuron call counts per batch.
Complexity: low-medium.

## Training mechanics

### New-neuron warmup

Optimizer sync copies the reference param group's hyperparameters exactly; fresh
neurons can destabilize established routes. Apply an lr multiplier ramp (or N-step
freeze) to newly added param groups. Complexity: low.

### Gradient checkpointing per route step

Activation memory scales with `max_steps` × fan-out. Wrap each recurrent route
step in `torch.utils.checkpoint`. Complexity: medium.

### Neuron lineage metadata

Record birth step + parent coordinate per grown neuron, serialized with the
checkpoint. Cheap to capture now, impossible to reconstruct later. Useful for
analyzing growth dynamics alongside the monitor's utilization heatmaps.
Complexity: low.

## Config ergonomics

### Depth-dependent neuron config

A single `neuron_config` is cloned for every coordinate. Allow per-z-layer
overrides (e.g. wider nucleus deeper) via the existing override machinery.
Complexity: medium.

### Neighborhood shape options

Terminal connections are a cartesian box (`xy_axis_range` × `z_axis_range`).
Alternative patterns (cross, sphere) as a `TerminalConnectionShapeOptions` enum,
matching the existing options idiom. Complexity: low-medium.

---

Suggested order by leverage: growth budget/cooldown → new-neuron warmup →
synchronized distributed growth; the rest as experiments demand.
