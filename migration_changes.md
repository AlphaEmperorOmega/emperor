# Module Migration Checklist

Reference pattern derived from `emperor/augmentations/adaptive_parameters/`
and `emperor/linears/`. Apply this checklist when migrating any module
(e.g. `emperor/attention/`, `emperor/parametric/`, future modules) into
the `core/` build/config/registry pattern.

Canonical reference impls:
- `emperor/augmentations/adaptive_parameters/`
- `emperor/linears/`
- `emperor/experts/` (in-progress reference)

---

## 1. Package layout

```
emperor/<module>/
  __init__.py            # flat public API + __all__
  options.py             # enums only
  config.py              # top-level module config (if needed)
  model.py               # top-level orchestrator (if needed)
  core/
    __init__.py          # re-exports of core public symbols
    config.py            # *Config dataclasses
    layers.py            # runtime Module classes
    _validator.py        # *Validator(ValidatorBase) classes
    monitor.py           # Lightning *MonitorCallback (optional)
    options.py           # core-only enums (if split from module options.py)
```

No `utils/` package in the migrated form. Move everything that used to live
in `<module>/utils/*` into `<module>/core/*` and delete (or temporarily shim)
the old `utils/` directory.

---

## 2. Config classes

Every config must:

1. Subclass `ConfigBase` from `emperor.base.utils`.
2. Be a `@dataclass`.
3. Use `optional_field("description")` for every field — no raw
   `dataclasses.field(...)` and no positional defaults.
4. Implement `_registry_owner(self) -> type` returning the runtime class
   the config builds. Import the runtime class lazily inside the method
   to avoid circular imports.
5. Use `TYPE_CHECKING` imports for forward references in field types.

Reference: `emperor/linears/core/config.py:13`,
`emperor/augmentations/adaptive_parameters/config.py:19`.

### 2a. Polymorphic dispatch

Replace "enum-on-base-config selects implementation" with **leaf config
subclasses**, one per variant. Each leaf overrides `_registry_owner()`.

Example: `DynamicWeightConfig` (base) →
`SingleModelDynamicWeightConfig`, `DualModelDynamicWeightConfig`,
`LowRankDynamicWeightConfig`, etc. Each owns one runtime class.

Do **not** keep a `model_type: SomeEnum` field on a god config to switch
behaviour.

Pre-migration modules may still hold enum-on-base configs; they become
leaves as part of their module's migration. The rule is mandatory for
**new** migration work, not retroactive debt collection.

### 2b. Drop redundant flags

If a flag is already expressed by a related config or by the runtime
layer, do not redeclare it on the dataclass.
Example: `DynamicBiasConfig.bias_flag` was removed because
`LinearLayerConfig.bias_flag` already governs bias presence.

---

## 3. Runtime modules

Every runtime class must:

1. Subclass `Module` from `emperor.base.utils`.
2. Constructor signature:
   ```python
   def __init__(self, cfg, overrides=None):
       super().__init__()
       config = getattr(cfg, "<module>_config", cfg)
       self.cfg = self._override_config(config, overrides)
   ```
   The `getattr(cfg, "<...>_config", cfg)` step preserves the
   `ModelConfig` compatibility path.
3. Run validation **before** building children:
   ```python
   <Module>Validator.validate(self)
   ```
4. Build children via `.build()` everywhere possible. Do **not** call
   runtime constructors directly inside the orchestrator:
   ```python
   # bad
   self.sampler = SamplerModel(sampler_config)
   # good
   self.sampler = self.sampler_config.build(sampler_overrides)
   ```
5. Method name for the main computation is `forward`. Not
   `_run_generator`, not `compute`, not custom names.
6. Internal model attributes named uniformly `self.model`, or
   `self.<role>_model` (e.g. `self.weight_model`, `self.bias_model`).
7. Use device-correct zero scalars: `inputs.new_zeros(())` not
   `torch.tensor(0.0)`. Required for CPU/CUDA/MPS portability.

Reference: `emperor/linears/core/layers.py:17`,
`emperor/augmentations/adaptive_parameters/model.py:15`.

---

## 4. Validators

Each module owns one `<Module>Validator(ValidatorBase)` class in
`core/_validator.py`.

Required pieces:

1. `OPTIONAL_FIELDS: set[str]` class attribute naming any config fields
   that may legitimately be `None`.
2. `@staticmethod validate(model)` entry point that runs:
   - `validate_required_fields(model.cfg)`
   - `validate_field_types(model.cfg)`
   - any module-specific structural checks
3. `@staticmethod validate_forward_inputs(...)` called as the first line
   of `Module.forward`. Validation happens at the forward boundary, not
   in helper methods.
4. No `_run_generator`-style helpers; validation is a flat list of
   static methods.

Delete any validator method that has zero callsites after a refactor.
Grep before assuming a validator is alive.

Reference: `emperor/linears/core/_validator.py:11`,
`emperor/experts/core/_validator.py:20`.

---

## 5. Naming sweep

Apply these renames consistently across any module being migrated.
Status column reflects repo state as of last sweep — verify with grep
before assuming clean.

| Old | New | Status |
|---|---|---|
| `_overwrite_config` | `_override_config` | clean |
| `LinearBase` | `LinearAbstract` | clean |
| `behaviours/` package | `augmentations/` | clean |
| `LinearMemory` enum/options | `DynamicMemory` | **stale**: `models/experts/presets.py`, `models/parametric_vector/presets.py`, `models/parametric_generator/presets.py` |
| enum value `NONE` | `DISABLED` | **stale**: `ClipParameterOptions.NONE` at `models/parametric_generator/config.py:92` and `emperor/parametric/core/mixtures/types/utils/_validator.py:54` |
| `<X>LayerStack` (custom stack) | use generic `LayerStack` where possible | **stale**: `ParametricLayerStack` at `emperor/parametric/core/stack.py:12` (handled by parametric migration). `DepthMappingLayerStack` in `adaptive_parameters/core/depth_mapper.py` is intentional (domain-specific subclass), not debt. |
| `utils/` (module-internal) | `core/` | **stale**: `emperor/attention/utils/`, `emperor/transformer/utils/`, `emperor/halting/utils/` |

---

## 6. Public API exposure

`emperor/<module>/__init__.py` re-exports the flat public surface and
declares `__all__`. Consumers should import from the package root, not
from `core/` submodules.

`emperor/<module>/core/__init__.py` re-exports core-level symbols for
intra-package use.

Reference: `emperor/augmentations/adaptive_parameters/__init__.py`.

---

## 7. Monitor callbacks

If the module exposes runtime metrics, add
`emperor/<module>/core/monitor.py` containing a
`lightning.pytorch.callbacks.Callback` subclass named
`<Module>MonitorCallback`. Wire it into the corresponding adaptive
trainer config (e.g. `linear_adaptive`).

Reference: `emperor/linears/core/monitor.py:10`,
`emperor/augmentations/adaptive_parameters/core/monitor.py`.

---

## 8. Memory state

State that previously lived on augmentations (e.g. `LinearMemory`)
belongs on the base `Layer` (in `emperor/base/layer/`). Do not
reintroduce per-augmentation memory buffers.

---

## 9. Verification steps for each migration

1. `python3 -m py_compile` on every changed file.
2. Run focused tests for the migrated module.
3. Run broader tests that import the module through `emperor/config.py`
   and `emperor/model.py`.
4. Grep for old paths and confirm only shims remain (or none, if shims
   were removed in the same pass).
5. Grep for direct runtime-class constructors that should now go through
   `.build()`.
6. Grep for `torch.tensor(0.0)`, `_overwrite_config`, `NONE` enum values,
   and module-specific renames listed in §5.

---

## 10. Per-module status

| Module | Status | Notes |
|---|---|---|
| `augmentations/adaptive_parameters` | done | reference impl |
| `linears` | done | reference impl |
| `experts` | done | routing refactor landed on `dev`; all three cleanups applied (dead `__maybe_*` removed from `model.py`, dead `validate_no_sampler_with_indices` removed from `_validator.py`, `layers.py` sampler built via `sampler_config.build(...)`). Tests green (37/38; sole failure is a CUDA-arch env mismatch in `test_forward_zero_loss_uses_input_device`, unrelated to experts code). |
| `parametric` | planned | see `plan.md` |
| `attention` | done | `core/` + leaf subpackages (`self_attention/`, `independent_attention/`, `mixture_of_attention_heads/`) all landed; `MultiHeadAttentionAbstract` base + leaf configs with own `_registry_owner`. |
| `sampler` | partially done | layout/configs/naming/public API clean. Debt: (1) §2a — `SamplerSparse`/`SamplerTopk`/`SamplerFull` share one `SamplerConfig`, dispatched by `top_k` value in `SamplerModel.__init_sampler_model`; split into `SparseSamplerConfig`/`TopkSamplerConfig`/`FullSamplerConfig`. (2) §3 rule 4 — direct constructor calls in `__init_sampler_model`; replace with `.build()` after split. (3) §3 rule 5 — main method named `get_probabilities_and_indices` not `forward`. |
| `base/layer` | done | host of generic `LayerStack`, `LayerConfig` |
| `transformer` | done | 5-leaf config split (coordinator + 2 stacks + 2 layers), siblings (no `TransformerLayerBase`), wrappers dropped (`Self/Cross/FeedForwardLayer` incompatible with new `Layer` base), `transformer/utils/` removed, `transformer/options.py` (dead enum) removed. `feed_forward` migrated as nested submodule with delegated `stack_config` dispatch. Causal-mask math (`_is_attention_mask_causal`, `__generate_causal_mask`, `causal_attention_mask_flag`, `source/target_sequence_length`) preserved on stacks as redundant copy of attention's authoritative version — user to clean up later. ViT-specific `TransformerEncoderModel`/`VITExperimentConfig` already mirrored in `models/vit/` (duplicates removed via `utils/` deletion). Legacy presets (`transformer/utils/presets.py`) removed; `models/vit/presets.py` and `models/bert/presets.py` left as pre-broken legacy out of migration scope. |
| `patch` | done | promoted to top-level `emperor/patch/` peer. §1 layout (`core/config.py` + `core/layers.py` + `core/_validator.py`). §2a leaf-config split: `LinearPatchEmbeddingConfig` / `ConvPatchEmbeddingConfig` siblings off abstract `PatchConfig`, each owns `_registry_owner`. `PatchOptions` enum + `PatchSelector` deleted. §3 rule 2 compat path (`getattr(cfg, "patch_config", cfg)`) restored; `main_cfg` reach-up dropped in favour of `LinearPatchEmbeddingConfig.embedding_stack_config` field. Conv variant flows through `LayerStack` → `Layer` → `Conv2dLayer` (new `emperor/convs/`): conv stack policy lives in `ConvPatchEmbeddingConfig.conv_stack_config: LayerStackConfig`, with the patch-extraction final layer driven by `LayerStackConfig.last_layer_overrides` (kernel=stride=patch_size); intermediates inherit the user-provided base `layer_config`. `PatchValidator` covers required-fields, types, positive-int dims, dropout-prob range, and 4D+channel forward-input checks. Pre-broken legacy callers (`models/vit/presets.py`, `docs/test_transformer_patch.py`) left out of migration scope — they depend on deleted transformer/attention preset chain. |
| `convs` | done | new top-level peer mirroring `emperor/linears/`. §1 layout (`core/config.py` + `core/layers.py` + `core/_validator.py`). `Conv2dLayer` wraps `nn.Conv2d` and plugs into `Layer.layer_model_config` (sibling of `LinearLayerConfig`). `Conv2dLayerValidator` covers required-fields, types, dims, kernel/stride/padding ranges, and 4D forward-input shape. Consumed by `emperor/patch/` conv variant; reusable by future CNN backbones. |
| `base/layer` (revisions) | done | `LayerStackConfig.last_layer_overrides: LayerConfig \| None` hook added — merged into the output layer's config via `LayerStack.__add_output_layer` → `__resolve_output_layer_overrides` (order: apply_output_pipeline_flag base → bias_overrides → last_layer_overrides). `LayerValidator` gains two spatial-aware checks: (1) `__validate_layer_norm_with_spatial_model` blocks `layer_norm_position != DISABLED` when `layer_model_config` has a `kernel_size` attr (duck-typed; covers Conv2dLayerConfig), and (2) `__validate_residual_with_strided_model` blocks `residual_flag=True` when `layer_model_config.stride > 1`. Both checks are duck-typed so no upward coupling from `base/layer` to `convs`. |

Update this table as each module lands.
