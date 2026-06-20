# Model Migration Guide

Use this guide to migrate an Emperor model package under `models/<group>/<name>/`
to the modern package and class structure. Treat `models/linears/linear` as the
canonical boundary-classifier archetype.

The migration goal is not only to rename files. The goal is to separate public
configuration, nested Emperor config construction, runtime model execution,
presets, and tests so every model package can be inspected, overridden, searched,
trained, and validated consistently.

## Canonical Archetype

Use these files in `models/linears/linear` as the source of truth:

- `config.py`: public constants, dataset options, trainer defaults, monitor
  options, and `SEARCH_SPACE_*` axes.
- `experiment_config.py`: dataclass payload for nested model config objects.
- `config_builder.py`: flat public kwargs converted into `ModelConfig`.
- `_control_config_factory.py`: hidden stack and controller assembly.
- `_controller_stack.py`: reusable controller stack option resolution.
- `model.py`: thin runtime module construction and forward path.
- `presets.py`: `ExperimentPreset`, `_PRESET_OVERRIDES`,
  `ExperimentPresets.PRESET_OVERRIDES`, `ExperimentPresets.PRESET_LOCKS`, and
  `Experiment`.
- `__main__.py`: CLI entry point through `models.parser`.
- `__init__.py`: public package exports.
- `test_model.py`: package-level contract tests.

## First Command

Always inspect the source package before editing:

```bash
python .agents/skills/emperor-model-factory/scripts/inspect_model_package.py \
  --model <group>/<name> \
  --pretty
```

Record the current `pattern`, `profile`, `builder_signature`,
`experiment_config_fields`, `preset_enum`, `preset_names`, `preset_locks`, and
`missing_modern_files`. Do not start by copying files blindly.

## Profile Classification

Classify the package before choosing a migration path.

| Profile | Modern target |
| --- | --- |
| `boundary_classifier` | Use the full `linears/linear` contract. |
| `modern_builder` wrapper | Preserve wrapper semantics; use boundary configs as wrapped source edges. |
| `encoder_classifier` | Keep encoder-specific fields; only copy the modern packaging style. |
| `pretraining_text` | Keep text/pretraining fields; only copy the modern packaging style. |
| `legacy_inline` | Split inline config/model/preset logic into modern files. |

The factory scaffold is currently canonical for `boundary_classifier` packages.
For encoder and pretraining models, migrate the folder/class shape without
forcing the three-field boundary classifier payload.

## Target Files

Every modern package should contain:

```text
models/<group>/<name>/
  __init__.py
  __main__.py
  config.py
  experiment_config.py
  config_builder.py
  model.py
  presets.py
  test_model.py
```

Add these helpers when the package has gate, halting, memory, recurrence,
sampler, router, adapter, or similarly complex controller construction:

```text
  _control_config_factory.py
  _controller_stack.py
```

## Boundary Classifier Contract

For a `boundary_classifier`, the modern `ExperimentConfig` has exactly these
task-level components:

```python
input_model_config
model_config
output_model_config
```

The runtime flow is:

1. Flatten image input.
2. Run input boundary projection from `input_dim` to `hidden_dim`.
3. Run reusable hidden block from `hidden_dim` to `hidden_dim`.
4. Run output boundary projection from `hidden_dim` to `output_dim`.
5. Return logits, or `(logits, auxiliary_loss)` when the hidden block returns
   auxiliary loss.

Boundary projections are task edges. Keep them outside hidden blocks, wrappers,
adapters, recurrent blocks, and reusable nucleus payloads.

## File-by-File Migration

### `config.py`

Move all public defaults into uppercase constants.

Rules:

- Keep dataset, trainer, callback, model, controller, search, and monitor
  sections easy to scan.
- Add a public constant for every public builder kwarg.
- Builder kwargs should normally be lowercase versions of constants.
- Add `SEARCH_SPACE_*` only for values that should be swept.
- Keep dataset options and trainer defaults in this file.
- Add monitor options here when the package exposes dashboard/logging callbacks.
- Update `models/config_overrides.py` aliases only when a public constant does
  not lower-case to the builder kwarg. The alias must work for config
  overrides, `--search-set`, `SEARCH_SPACE_*` extraction, and explicit
  `search_keys`. Add a regression test whenever a public search axis maps to a
  differently named builder kwarg, for example `WEIGHT_GENERATOR_DEPTH` mapping
  to `generator_depth`.
- For adaptive parameter packages, keep shared hidden-block generator defaults
  in `ADAPTIVE_STACK_*`. Put component-specific generator stack controls under
  their owning option sections (`WEIGHT_*`, `BIAS_*`, `DIAGONAL_*`, `MASK_*`),
  not in a separate mixed block.
- Adaptive hidden-block components must have explicit component enable flags
  next to their option constants. Use `WEIGHT_OPTION_FLAG`,
  `BIAS_OPTION_FLAG`, `DIAGONAL_OPTION_FLAG`, and `MASK_OPTION_FLAG`. The mask
  implementation option remains `ROW_MASK_OPTION`; the enable flag is still
  `MASK_OPTION_FLAG` because it controls the Mask component section.
- Do not add boundary projector option flags during a hidden-block adaptive
  migration unless that migration explicitly changes boundary projector
  semantics.
- Component generator stack overrides should be hidden-block-only unless the
  migration explicitly adds boundary projector controls. If the component
  independent flag is false, leave the component config `model_config=None` so
  runtime inheritance from `AdaptiveParameterAugmentationConfig.model_config`
  is preserved.
- Viewer full-configuration accordions are derived from the config schema. Use
  recognized section headings and consistent stack names so no frontend
  special case is needed. Stack child accordions are detected from fields such
  as `*_STACK_INDEPENDENT_FLAG`, `*_STACK_NUM_LAYERS`,
  `*_STACK_ACTIVATION`, plus stack-scoped fields such as `*_HIDDEN_DIM`,
  `*_LAYER_NORM_POSITION`, and `*_BIAS_FLAG` when the section itself is a stack
  section.

### `experiment_config.py`

Create a dataclass extending `ConfigBase`.

For boundary classifiers:

```python
@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: LayerConfig | None = optional_field(...)
    model_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(...)
    output_model_config: LayerConfig | None = optional_field(...)
```

For other profiles, keep the profile-specific fields. Examples:

- Encoder classifier: `patch_config`, `positional_embedding_config`,
  `encoder_config`, `output_config`.
- Text pretraining: `positional_embedding_config`,
  `embedding_dropout_probability`, `encoder_config`, task heads.
- Wrapper: source input config, adapter/wrapper config, source output config.

### `config_builder.py`

Create one public builder class named `<ModelName>ConfigBuilder`.

Rules:

- Constructor kwargs are flat public values, not nested config objects.
- Defaults come from `config.py`.
- Store normalized builder state on `self`.
- `build()` returns `emperor.config.ModelConfig`.
- `build()` constructs the task-edge configs and delegates complex hidden
  controller assembly to helpers.
- Preserve dataset boundary overrides by passing `input_dim` and `output_dim`
  into `ModelConfig`.
- Avoid embedding preset-specific logic in the builder. Presets provide flat
  kwargs; the builder builds whatever those kwargs request.
- Treat adaptive component option flags as authoritative. If an adaptive
  component flag is false, return no component config even when its option is
  set. If the flag is true and the option is missing, raise a clear
  `ValueError`. If both are set, build the component normally.

Boundary classifier `build()` should follow this shape:

```python
input_model_config = ...
model_config = ControlConfigFactory(self).build()
output_model_config = ...

return ModelConfig(
    learning_rate=self.learning_rate,
    batch_size=self.batch_size,
    input_dim=self.input_dim,
    hidden_dim=self.hidden_dim,
    output_dim=self.output_dim,
    experiment_config=ExperimentConfig(
        input_model_config=input_model_config,
        model_config=model_config,
        output_model_config=output_model_config,
    ),
)
```

### `_controller_stack.py`

Use this helper when several controllers need similar stack options.

Rules:

- Define a source dataclass for optional controller overrides.
- Define a resolved options dataclass with concrete values.
- Add a resolver that returns shared defaults unless an independent flag is
  enabled.
- Build the concrete controller stack from resolved options.

Important behavior:

- Controller-specific overrides should not apply unless the matching
  `*_stack_independent_flag` is true.
- If independent mode is false, inherit submodule defaults.

### `_control_config_factory.py`

Use this helper when hidden block construction has controller complexity.

Rules:

- Accept the builder in `__init__`.
- `build()` returns the hidden reusable config payload.
- Build per-layer gate, halting, memory, recurrent gate, recurrent halting, and
  shared memory config here.
- Keep recurrent wrapping here, not in `model.py`.
- Validate contradictory options here, for example `shared_gate_config` together
  with enabled per-layer gate config.
- Keep boundary input/output projections out of this helper.

### `model.py`

Keep the model thin. It should construct modules from `ExperimentConfig` and run
the forward path. It should not know about presets, search, CLI flags, or config
normalization.

Boundary classifier rules:

- Extend `ClassifierExperiment`.
- Store `cfg`, `cfg.experiment_config`, and built modules.
- Build modules by calling the config object's build method with input/output
  dimension overrides.
- Flatten image input in `forward()`.
- Use `Layer.run_model_returning_hidden` for boundary projections.
- Use `Layer.run_model_returning_state` for hidden blocks that may return
  `LayerState`.
- Propagate auxiliary loss when present.

### `presets.py`

Use `ExperimentPreset`, not `ExperimentOptions`, for migrated packages unless
preserving an older external API is explicitly required.

Rules:

- Define enum values as user-facing preset descriptions.
- Define `_PRESET_OVERRIDES` mapping each preset to flat builder kwargs.
- Expose `_PRESET_OVERRIDES` as `ExperimentPresets.PRESET_OVERRIDES`.
- Generate `ExperimentPresets.PRESET_LOCKS` from the same preset-owned behavior.
- Keep preset methods thin; complex config construction belongs in the builder
  and helpers.
- Rejected overrides should be handled by the shared preset/config override
  machinery through `PRESET_LOCKS`.
- For adaptive component presets, add the matching option flag whenever a
  preset sets a component option, and lock both the option and the flag. For
  example, `row_mask_option` implies `mask_option_flag=True`.

Preferred shape:

```python
_PRESET_OVERRIDES = {
    ExperimentPreset.BASELINE: {},
    ExperimentPreset.GATING: {"stack_gate_flag": True},
}

class ExperimentPresets(ExperimentPresetsBase):
    PRESET_OVERRIDES = _PRESET_OVERRIDES
    PRESET_LOCKS = _preset_locks(PRESET_OVERRIDES)

    def get_config(...):
        preset_callback = self._preset_callback_for_preset(model_config_preset)
        return self._create_preset_search_space_configs(...)

    def _preset_for_preset(self, preset, **kwargs):
        return self._preset(**{**kwargs, **self.PRESET_OVERRIDES[preset]})

    def _preset(self, **kwargs):
        return ModelConfigBuilder(**kwargs).build()
```

### `__main__.py`

Use the shared parser entry point:

```python
from models.parser import get_experiment_parser, resolve_dataset_names, resolve_experiment_mode
from models.<group>.<name> import Experiment, ExperimentPreset

EXPERIMENT_MODULE_PATH = "models.<group>.<name>"
```

Parse args, resolve mode, construct `Experiment(mode.preset)`, and call
`train_model(...)` with search, overrides, logdir, and selected datasets.

### `__init__.py`

Export the public API:

```python
from models.<group>.<name>.presets import Experiment, ExperimentPreset

__all__ = ["Experiment", "ExperimentPreset"]
```

### `test_model.py`

Tests should prove the package contract, not implementation trivia.

Required coverage:

- Public imports expose `Experiment` and `ExperimentPreset`.
- CLI entry point resolves without starting training.
- Every preset builds config and forwards one fake batch.
- Baseline forwards every dataset in `DATASET_OPTIONS` for image classifiers.
- Shape checks use `dataset.num_classes`.
- Search mode still creates configs.
- Unknown search axes fail.
- Search keys whose public config names require aliases reach the intended
  builder kwargs and nested config fields.
- Preset-owned behavior appears in `PRESET_LOCKS`.
- Locked preset config/search overrides are rejected.
- Unlocked overrides still work.
- New controller and adaptive generator-stack behavior is asserted at the nested
  config level.
- Adaptive component flags are asserted explicitly: defaults produce no
  weight/bias/diagonal/mask components, option-without-flag is ignored,
  flag-without-option raises, and flag-plus-option builds the expected
  component.
- Viewer/schema tests should assert that component flag fields appear in the
  correct sections, that `MASK_OPTION_FLAG` is exposed as `mask_option_flag`,
  and that stack-like config fields render as inner accordions without adding a
  one-off frontend mapping for the model.
- Hidden blocks that return auxiliary loss are forwarded as `(logits, loss)`.

## Migration Algorithm

Follow this sequence for each package.

1. Inspect current package:

   ```bash
   python .agents/skills/emperor-model-factory/scripts/inspect_model_package.py \
     --model <group>/<name> \
     --pretty
   ```

2. Classify the profile from `experiment_config_fields`.

3. If the package is legacy inline, split it into `config.py`,
   `experiment_config.py`, `config_builder.py`, `model.py`, `presets.py`,
   `__main__.py`, `__init__.py`, and `test_model.py`.

4. If the package is already modern, keep existing behavior and migrate only the
   missing contract pieces:

   - Add `PRESET_OVERRIDES` if absent.
   - Generate `PRESET_LOCKS` from preset overrides.
   - Extract controller construction helpers when the builder is hard to scan.
   - Add missing tests for search, locks, and nested config wiring.

5. For boundary classifiers, make `models/linears/linear` the shape reference:

   - `ExperimentConfig` has input, hidden, and output configs.
   - `Model` is a thin three-stage classifier.
   - Boundary projections stay outside the hidden block.

6. For wrappers, preserve source boundary configs and wrap only the source
   hidden block. Do not put source input/output projections inside the adapter.

7. For encoder/text profiles, preserve their domain-specific config payloads.
   Adopt the modern package and preset structure, but do not force the
   boundary-classifier three-field shape.

8. Run validation and tests.

9. Re-run inspector and compare before/after summary.

## Validation Commands

For a migrated package:

```bash
python .agents/skills/emperor-model-factory/scripts/inspect_model_package.py \
  --model <group>/<name> \
  --pretty

python .agents/skills/emperor-model-factory/scripts/validate_model_package.py \
  --model <group>/<name> \
  --all-presets

python -m unittest models.<group>.<name>.test_model
```

When the package supports generated boundary-classifier validation, the modern
contract should pass. For inspection-only profiles, validation may still import,
build, forward, and graph supported presets, but the factory does not scaffold
those profiles yet.

## Current Migration Priority

As of this guide, package status is:

| Package | Status | Suggested action |
| --- | --- | --- |
| `linears/linear` | Canonical modern boundary classifier | Keep as archetype. |
| `linears/linear_adaptive` | Modern boundary classifier | Keep adaptive component flags, hidden-block generator stack inheritance, boundary projector semantics, and aliased search axes covered by tests. |
| `experts/experts_linear` | Modern boundary classifier | Extract controller/router helpers and add `PRESET_OVERRIDES`. |
| `experts/experts_linear_adaptive` | Modern boundary classifier | Extract helpers; preserve adaptive expert behavior. |
| `neuron/neuron_linear` | Modern wrapper | Preserve wrapper semantics; mirror source presets/locks. |
| `parametric/*` | Legacy inline | Split into modern package files. |
| `transformer_encoder/bert_linear` | Modern pretraining text | Keep profile fields; align preset structure only. |
| `transformer_encoder/vit_linear` | Modern encoder classifier | Keep profile fields; align preset structure only. |

## Done Criteria

A migration is complete when:

- Inspector reports no missing required modern files.
- Public imports still work.
- CLI mode resolution still works.
- Presets are represented by `ExperimentPreset`.
- `ExperimentPresets.PRESET_OVERRIDES` exists.
- `ExperimentPresets.PRESET_LOCKS` prevents contradictory preset overrides.
- Builder kwargs are flat and backed by constants.
- Public config/search aliases resolve to builder kwargs before configs are
  constructed.
- Boundary classifier task edges are outside hidden reusable blocks.
- Every preset forwards one fake batch.
- Baseline forwards every configured image dataset.
- Unit tests pass.
- Validation passes for all supported presets.
