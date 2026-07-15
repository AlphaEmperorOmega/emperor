"""Traceability manifest for the complete ``emperor.attention`` package."""


def class_contract(responsibilities, test_module):
    return {
        "responsibilities": tuple(responsibilities),
        "tests": (test_module.removeprefix("tests."),),
    }


def module_contract(responsibilities, test_module, **classes):
    return {
        "responsibilities": tuple(responsibilities),
        "tests": (test_module.removeprefix("tests."),),
        "classes": classes,
    }


EXPORT_TEST = "tests.unit.test_attention_contracts.TestAttentionExports"

ATTENTION_CONTRACT_MANIFEST = {
    "emperor.attention": module_contract(("exports",), EXPORT_TEST),
    "emperor.attention.core": module_contract(("exports",), EXPORT_TEST),
    "emperor.attention.core._validator": module_contract(
        ("configuration validation", "runtime errors", "static contracts"),
        "tests.unit.test_attention_validator",
        AttentionValidatorBase=class_contract(
            ("shape", "mask", "divisibility", "static projection validation"),
            "tests.unit.test_attention_validator",
        ),
        MultiHeadAttentionValidator=class_contract(
            ("configuration", "runtime tensor", "selected source validation"),
            "tests.unit.test_attention_correctness_regressions",
        ),
    ),
    "emperor.attention.core.config": module_contract(
        ("configuration schema", "build dispatch", "runtime maxima documentation"),
        "tests.unit.test_attention",
        MultiHeadAttentionConfig=class_contract(
            ("defaults", "overrides", "abstract build"),
            "tests.unit.test_attention",
        ),
    ),
    "emperor.attention.core.handlers": module_contract(
        ("private namespace",), EXPORT_TEST
    ),
    "emperor.attention.core.handlers.batch": module_contract(
        ("layout inference", "identity preservation", "output restoration"),
        "tests.unit.test_attention_batch_dimension_manager",
        BatchDimensionManager=class_contract(
            (
                "unbatched",
                "sequence-first",
                "batch-first",
                "runtime-state restoration",
            ),
            "tests.unit.test_attention_batch_dimension_manager",
        ),
    ),
    "emperor.attention.core.handlers.bias": module_contract(
        ("source extension", "neutral mask padding", "bias gradients"),
        "tests.unit.test_attention_key_value_bias",
        KeyValueBias=class_contract(
            ("disabled identity", "head-layout append", "gradient flow"),
            "tests.unit.test_attention_key_value_bias",
        ),
    ),
    "emperor.attention.core.handlers.mask": module_contract(
        ("validation", "canonicalization", "causal generation", "mask merge"),
        "tests.unit.test_attention_mask",
        Mask=class_contract(
            ("bool masks", "additive masks", "rectangular masks", "no mutation"),
            "tests.unit.test_attention_mask",
        ),
    ),
    "emperor.attention.core.handlers.processor": module_contract(
        ("relative logits", "output projection", "abstract dispatch"),
        "tests.unit.test_attention_processor",
        ProcessorBase=class_contract(
            ("dimension resolution", "relative source topology", "abstract guard"),
            "tests.unit.test_attention_processor",
        ),
    ),
    "emperor.attention.core.handlers.projector": module_contract(
        ("projection construction", "state loss accumulation", "abstract dispatch"),
        "tests.unit.test_attention_projector",
        ProjectorBase=class_contract(
            ("QKV dimensions", "output projection", "auxiliary loss lifecycle"),
            "tests.unit.test_attention_projector",
        ),
    ),
    "emperor.attention.core.handlers.reshaper": module_contract(
        ("head layouts", "static replacement", "abstract dispatch"),
        "tests.unit.test_attention_contracts",
        ReshaperBase=class_contract(
            ("head dimensions", "identity pre-attention path", "abstract guard"),
            "tests.unit.test_attention_contracts",
        ),
        AttentionReshaper=class_contract(
            ("query reshape", "independent key/value widths", "static selection"),
            "tests.unit.test_attention_projector",
        ),
    ),
    "emperor.attention.core.handlers.zero_attention": module_contract(
        ("zero source extension", "neutral mask padding"),
        "tests.unit.test_attention_zero_attention",
        ZeroAttention=class_contract(
            ("disabled identity", "head-layout append", "dtype and device"),
            "tests.unit.test_attention_zero_attention",
        ),
    ),
    "emperor.attention.core.layers": module_contract(
        ("attention pipeline", "source topology", "public output contract"),
        "tests.unit.test_attention",
        MultiHeadAttentionAbstract=class_contract(
            ("construction", "validation", "forward orchestration", "abstract guard"),
            "tests.unit.test_attention_contracts",
        ),
    ),
    "emperor.attention.core.monitor": module_contract(
        ("instrumentation lifecycle", "metrics", "logging fallbacks", "history"),
        "tests.unit.test_attention_monitor",
        _AttentionObservation=class_contract(
            ("typed detached forward-pass values",),
            "tests.unit.test_attention_monitor",
        ),
        _AttentionDiagnosticMetrics=class_contract(
            ("typed calculator output",),
            "tests.unit.test_attention_monitor",
        ),
        _AttentionTrackingContext=class_contract(
            ("typed metric-emission context",),
            "tests.unit.test_attention_monitor",
        ),
        _AttentionMethodReplacement=class_contract(
            ("typed reversible method replacement",),
            "tests.unit.test_attention_monitor",
        ),
        _AttentionDiagnosticsTracker=class_contract(
            ("observation capture", "exact-weight priority", "detachment"),
            "tests.unit.test_attention_monitor",
        ),
        _AttentionDiagnosticsTrackerManager=class_contract(
            ("instrumentation attachment", "exact restoration", "cleanup"),
            "tests.unit.test_attention_monitor",
        ),
        _AttentionWeightAdapter=class_contract(
            ("legacy weight-layout compatibility",),
            "tests.unit.test_attention_monitor",
        ),
        _AttentionDiagnostics=class_contract(
            ("pure metric calculation", "approximate fallback", "mask handling"),
            "tests.unit.test_attention_monitor",
        ),
        AttentionMonitorCallback=class_contract(
            (
                "lifecycle",
                "cadence",
                "metric emission",
                "bounded history",
            ),
            "tests.unit.test_attention_monitor",
        ),
    ),
    "emperor.attention.core.runtime": module_contract(
        ("immutable runtime values", "runtime shape helpers"),
        "tests.unit.test_attention_runtime_values",
        QKV=class_contract(
            ("immutability", "tensor identity", "replacement"),
            "tests.unit.test_attention_runtime_values",
        ),
        AttentionMasks=class_contract(
            ("immutability", "optional masks", "replacement"),
            "tests.unit.test_attention_runtime_values",
        ),
        AttentionRuntimeShape=class_contract(
            ("branch count", "source extension", "real source length"),
            "tests.unit.test_attention_contracts",
        ),
    ),
    "emperor.attention.core.state": module_contract(
        ("layer-state mask preservation",),
        "tests.unit.test_attention_contracts",
        AttentionLayerState=class_contract(
            ("hidden state", "padding mask", "attention mask"),
            "tests.unit.test_attention_contracts",
        ),
    ),
    "emperor.attention.core.variants": module_contract(
        ("variant exports",), EXPORT_TEST
    ),
    "emperor.attention.core.variants.independent_attention": module_contract(
        ("independent variant exports",), EXPORT_TEST
    ),
    "emperor.attention.core.variants.independent_attention.config": module_contract(
        ("build dispatch",),
        "tests.unit.test_attention",
        IndependentAttentionConfig=class_contract(
            ("construction", "overrides", "build mapping"),
            "tests.unit.test_attention",
        ),
    ),
    "emperor.attention.core.variants.independent_attention.layer": module_contract(
        ("component wiring", "public forward"),
        "tests.unit.test_attention",
        IndependentAttention=class_contract(
            ("cross attention", "static source", "relative positioning"),
            "tests.unit.test_attention_correctness_regressions",
        ),
    ),
    "emperor.attention.core.variants.independent_attention.processor": module_contract(
        ("SDPA math", "mask layout", "relative additive bias", "dropout"),
        "tests.unit.test_attention_processor",
        IndependentProcessor=class_contract(
            ("exact math", "relative gradients", "mask combination"),
            "tests.unit.test_attention_correctness_regressions",
        ),
    ),
    "emperor.attention.core.variants.independent_attention.projector": module_contract(
        ("separate QKV projections",),
        "tests.unit.test_attention_projector",
        IndependentProjector=class_contract(
            ("unequal QK/value widths", "projection gradients"),
            "tests.unit.test_attention_projector",
        ),
    ),
    "emperor.attention.core.variants.independent_attention.validator": module_contract(
        ("cross-attention input validation",),
        "tests.unit.test_attention_validator",
        IndependentAttentionValidator=class_contract(
            ("weight restriction", "key/value compatibility"),
            "tests.unit.test_attention_validator",
        ),
    ),
    "emperor.attention.core.variants.mixture_of_attention_heads": module_contract(
        ("mixture variant exports",), EXPORT_TEST
    ),
    "emperor.attention.core.variants.mixture_of_attention_heads.bias": module_contract(
        ("shared and expert bias extension",),
        "tests.unit.test_attention_mixture_of_attention_heads",
        MixtureOfAttentionHeadsKeyValueBias=class_contract(
            ("shared branches", "expert branches"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    (
        "emperor.attention.core.variants.mixture_of_attention_heads.config"
    ): module_contract(
        ("expert configuration schema", "build dispatch"),
        "tests.unit.test_attention_mixture_of_attention_heads",
        MixtureOfAttentionHeadsConfig=class_contract(
            ("experts", "K/V expert mode", "build mapping"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    "emperor.attention.core.variants.mixture_of_attention_heads.layer": module_contract(
        ("mixture component wiring", "public forward"),
        "tests.unit.test_attention_mixture_of_attention_heads",
        MixtureOfAttentionHeads=class_contract(
            ("shared K/V", "expert K/V", "auxiliary loss"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    "emperor.attention.core.variants.mixture_of_attention_heads.mask": module_contract(
        ("expert mask normalization", "mask validation", "merge"),
        "tests.unit.test_attention_mixture_of_attention_heads",
        MixtureOfAttentionHeadsMask=class_contract(
            ("standard branches", "expert branches", "padding expansion"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    (
        "emperor.attention.core.variants.mixture_of_attention_heads.processor"
    ): module_contract(
        ("expert attention equations", "relative logits", "dropout"),
        "tests.unit.test_attention_processor",
        MixtureOfAttentionHeadsProcessor=class_contract(
            ("shared K/V math", "expert K/V math", "relative source topology"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    (
        "emperor.attention.core.variants.mixture_of_attention_heads.projector"
    ): module_contract(
        ("routing", "expert projections", "auxiliary loss"),
        "tests.unit.test_attention_projector",
        MixtureOfAttentionHeadsProjector=class_contract(
            ("sample once", "shared K/V", "expert K/V", "output reduction"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    (
        "emperor.attention.core.variants.mixture_of_attention_heads.reshaper"
    ): module_contract(
        ("expert/head axis restoration", "static replacement"),
        "tests.unit.test_attention_mixture_of_attention_heads",
        MixtureOfAttentionHeadsReshaper=class_contract(
            ("shared K/V layout", "expert K/V layout", "unsupported static expert K/V"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    (
        "emperor.attention.core.variants.mixture_of_attention_heads.validator"
    ): module_contract(
        ("expert configuration", "routing", "unsupported combinations"),
        "tests.unit.test_attention_validator",
        MixtureOfAttentionHeadsValidator=class_contract(
            ("nested types", "routing dimensions", "static expert K/V rejection"),
            "tests.unit.test_attention_validator",
        ),
    ),
    (
        "emperor.attention.core.variants.mixture_of_attention_heads.zero_attention"
    ): module_contract(
        ("shared and expert zero extension",),
        "tests.unit.test_attention_mixture_of_attention_heads",
        MixtureOfAttentionHeadsZeroAttention=class_contract(
            ("shared branches", "expert branches"),
            "tests.unit.test_attention_mixture_of_attention_heads",
        ),
    ),
    "emperor.attention.core.variants.self_attention": module_contract(
        ("self-attention variant exports",), EXPORT_TEST
    ),
    "emperor.attention.core.variants.self_attention.config": module_contract(
        ("projection strategy", "build dispatch"),
        "tests.unit.test_attention_projector",
        SelfAttentionProjectionStrategy=class_contract(
            ("fused mapping", "separate mapping"),
            "tests.unit.test_attention_projector",
        ),
        SelfAttentionConfig=class_contract(
            ("construction", "overrides", "build mapping"),
            "tests.unit.test_attention",
        ),
    ),
    "emperor.attention.core.variants.self_attention.layer": module_contract(
        ("self-attention component wiring", "public forward"),
        "tests.unit.test_attention",
        SelfAttention=class_contract(
            ("identity QKV", "returned weights", "source extensions"),
            "tests.unit.test_attention_correctness_regressions",
        ),
    ),
    "emperor.attention.core.variants.self_attention.processor": module_contract(
        ("scaled dot-product math", "weights", "relative logits", "dropout"),
        "tests.unit.test_attention_processor",
        SelfAttentionProcessor=class_contract(
            ("exact math", "weight formatting", "relative source topology"),
            "tests.unit.test_attention_processor",
        ),
    ),
    "emperor.attention.core.variants.self_attention.projector": module_contract(
        ("fused and separate projections", "output projection"),
        "tests.unit.test_attention_projector",
        SelfAttentionProjector=class_contract(
            ("fused split", "separate projections", "gradient flow"),
            "tests.unit.test_attention_projector",
        ),
    ),
    "emperor.attention.core.variants.self_attention.validator": module_contract(
        ("self-attention identity", "projection constraints"),
        "tests.unit.test_attention_validator",
        SelfAttentionValidator=class_contract(
            ("QKV identity", "equal dimensions", "recurrent restriction"),
            "tests.unit.test_attention_validator",
        ),
    ),
}


__all__ = ["ATTENTION_CONTRACT_MANIFEST"]
