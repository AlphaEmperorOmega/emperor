from __future__ import annotations

_DESCRIPTION_TEXT_BY_POLICY: dict[str, str] = {
    "Model": (
        "Top-level inspected model wrapper that owns the architecture, loss, "
        "metrics, and runtime modules for the selected preset."
    ),
    "ModuleList": (
        "Container that stores an ordered list of child modules; execution is "
        "defined by the parent module."
    ),
    "Sequential": (
        "Container that applies child modules in order, passing each output to "
        "the next child."
    ),
    "Dropout": (
        "Regularization module that randomly zeroes activations during training "
        "and is inactive during evaluation."
    ),
    "LayerNorm": (
        "Normalizes features within each sample to stabilize hidden-state "
        "scale before or after a layer block."
    ),
    "RecurrentIterationSchedule": (
        "Tracks forward-call progress, active recurrent depth, and the "
        "gradient-enabled transition suffix."
    ),
    "CrossEntropyLoss": ("Runtime loss module for multi-class classification targets."),
    "ClassifierMetricsLogger": (
        "Runtime module that groups classifier metrics for train, validation, "
        "and test reporting."
    ),
    "LanguageModelMetricsLogger": (
        "Runtime module that groups language-model metrics for train, "
        "validation, and test reporting."
    ),
    "SequenceClassifierMetricsLogger": (
        "Runtime module that groups sequence-classifier metrics for train, "
        "validation, and test reporting."
    ),
    "MulticlassAccuracy": (
        "Runtime metric that reports the share of classified examples whose "
        "predicted class matches the target class."
    ),
    "MulticlassF1Score": (
        "Runtime metric that reports the harmonic mean of classifier precision "
        "and recall across classes."
    ),
    "KeyValueBias": (
        "Internal attention helper that adds learned key/value bias terms."
    ),
    "SamplerAuxiliaryLosses": (
        "Internal mixture-of-experts helper that tracks auxiliary routing losses."
    ),
    "SelfAttentionProcessor": (
        "Internal attention helper that prepares attention inputs and masks."
    ),
    "SelfAttentionProjector": (
        "Internal attention helper that projects hidden states into attention "
        "query, key, and value tensors."
    ),
    "Unfold": (
        "Internal tensor reshaping module that extracts sliding local blocks "
        "from an input tensor."
    ),
    "LinearLayer": (
        "Applies a learned linear projection with configured input/output "
        "dimensions and optional bias."
    ),
    "LinearLayerConfig": (
        "Builds a learned linear projection with configured input/output "
        "dimensions and optional bias."
    ),
    "AdaptiveLinearLayer": (
        "Applies a learned linear projection that can optionally augment "
        "parameters from the current input."
    ),
    "AdaptiveLinearLayerConfig": (
        "Builds a linear projection that can optionally augment parameters from "
        "the current input."
    ),
    "Layer": (
        "Applies one configured layer block with optional activation, residuals, "
        "normalization, gating, halting, and memory hooks."
    ),
    "LayerConfig": (
        "Builds a Layer block with optional activation, residuals, normalization, "
        "gating, halting, and memory hooks."
    ),
    "LayerStack": (
        "Runs an ordered stack of Layer blocks, with shared dimensions and "
        "optional shared gate, halting, or memory modules."
    ),
    "LayerStackConfig": (
        "Builds an ordered stack of Layer blocks, with shared dimensions and "
        "optional shared gate, halting, or memory modules."
    ),
    "RecurrentLayer": (
        "Reuses a configured block for multiple recurrent steps, optionally "
        "adding recurrent gating, normalization, halting, or memory."
    ),
    "RecurrentLayerConfig": (
        "Builds a recurrent block that can run for multiple steps with optional "
        "gating, normalization, halting, or memory."
    ),
    "RecurrentCompositionConfig": (
        "Abstract recurrent composition Interface; use a concrete recurrent config."
    ),
    "TinyRecursiveModelRecurrent": (
        "Reuses one transition block for Tiny Recursive Model latent and answer "
        "updates across a fixed answer-update schedule."
    ),
    "TinyRecursiveModelRecurrentConfig": (
        "Builds Tiny Recursive Model recurrence with one shared transition, a "
        "latent-update count per answer update, and an answer-update count."
    ),
    "HierarchicalReasoningModelRecurrent": (
        "Runs distinct low- and high-level transitions on nested recurrent clocks."
    ),
    "HierarchicalReasoningModelRecurrentConfig": (
        "Builds Hierarchical Reasoning Model recurrence with separate low- and "
        "high-level transitions and clock counts."
    ),
    "MixtureOfExperts": (
        "Routes inputs across a set of expert modules using sampler probabilities "
        "and combines or maps the selected expert outputs."
    ),
    "MixtureOfExpertsMap": (
        "Routes inputs across experts and returns mapped expert outputs."
    ),
    "MixtureOfExpertsReduce": (
        "Routes inputs across experts and reduces selected expert outputs back "
        "into one representation."
    ),
    "MixtureOfExpertsLayer": (
        "Wraps mixture-of-experts routing in the standard Layer pipeline."
    ),
    "MixtureOfExpertsConfig": (
        "Configures expert count, routing, capacity, weighting, sampler behavior, "
        "and expert model construction."
    ),
    "MixtureOfExpertsLayerConfig": (
        "Builds a mixture-of-experts layer inside the standard Layer pipeline."
    ),
    "MixtureOfExpertsModelConfig": (
        "Builds a model around a mixture-of-experts layer stack."
    ),
    "LayerGate": (
        "Combines a learned gate output with the current layer value by scaling "
        "or addition."
    ),
    "GateConfig": (
        "Configures a layer gate network and how its output is composed with "
        "the current value."
    ),
    "AdditiveResidual": "Adds the current and previous hidden values.",
    "WeightedResidual": (
        "Adds the previous hidden value to a learned tanh-weighted current value."
    ),
    "WeightedBlendResidual": (
        "Convexly blends current and previous hidden values with a learned weight."
    ),
    "AttentionResidual": (
        "Routes across forward-local residual-depth sources with learned attention."
    ),
    "ResidualConfig": (
        "Abstract residual configuration Interface; use a concrete residual config."
    ),
    "AdditiveResidualConfig": "Builds direct additive residual composition.",
    "WeightedResidualConfig": (
        "Builds tanh-weighted residual composition with a scalar or generated "
        "coefficient."
    ),
    "WeightedBlendResidualConfig": (
        "Builds sigmoid convex residual blending with a scalar or generated "
        "coefficient."
    ),
    "AttentionResidualConfig": (
        "Builds learned attention routing across residual-depth history."
    ),
    "HaltingConfig": (
        "Configures adaptive computation halting thresholds, dropout, hidden-state "
        "mode, and gate network."
    ),
    "SoftHalting": (
        "Accumulates weighted recurrent states until the halting threshold is met."
    ),
    "SoftHaltingConfig": (
        "Builds soft halting, which accumulates weighted recurrent states until "
        "the threshold is met."
    ),
    "StickBreaking": (
        "Allocates remaining recurrent probability mass step by step until the "
        "halting threshold is met."
    ),
    "StickBreakingConfig": (
        "Builds stick-breaking halting, which allocates remaining recurrent "
        "probability mass over steps."
    ),
    "NeuronCluster": (
        "Maintains a 3D cluster of routed neurons that can traverse, branch, and "
        "grow during training."
    ),
    "NeuronClusterConfig": (
        "Configures a 3D routed neuron cluster, including capacity, traversal, "
        "sampling, and growth controls."
    ),
}

DESCRIPTION_POLICY_BY_TYPE_ID: dict[str, str] = {
    "torch.nn.modules.container.ModuleList": "ModuleList",
    "torch.nn.modules.container.Sequential": "Sequential",
    "torch.nn.modules.dropout.Dropout": "Dropout",
    "torch.nn.modules.normalization.LayerNorm": "LayerNorm",
    "emperor.layers._composition.recurrent.runtime.iteration_schedule.RecurrentIterationSchedule": "RecurrentIterationSchedule",
    "torch.nn.modules.loss.CrossEntropyLoss": "CrossEntropyLoss",
    "emperor.experiments.classifier._metrics.ClassifierMetricsLogger": "ClassifierMetricsLogger",
    "emperor.experiments.language_model._metrics.LanguageModelMetricsLogger": "LanguageModelMetricsLogger",
    "emperor.experiments.sequence_classifier._metrics.SequenceClassifierMetricsLogger": "SequenceClassifierMetricsLogger",
    "torchmetrics.classification.accuracy.MulticlassAccuracy": "MulticlassAccuracy",
    "torchmetrics.classification.f_beta.MulticlassF1Score": "MulticlassF1Score",
    "emperor.attention._ops.bias.KeyValueBias": "KeyValueBias",
    "emperor.sampler._selection.losses.SamplerAuxiliaryLosses": "SamplerAuxiliaryLosses",
    "emperor.attention._variants.self_attention.processing.SelfAttentionProcessor": "SelfAttentionProcessor",
    "emperor.attention._variants.self_attention.projection.SelfAttentionProjector": "SelfAttentionProjector",
    "torch.nn.modules.fold.Unfold": "Unfold",
    "emperor.linears._layer.LinearLayer": "LinearLayer",
    "emperor.linears._config.LinearLayerConfig": "LinearLayerConfig",
    "emperor.augmentations.adaptive_parameters._linear_adapter.AdaptiveLinearLayer": "AdaptiveLinearLayer",
    "emperor.augmentations.adaptive_parameters._config.AdaptiveLinearLayerConfig": "AdaptiveLinearLayerConfig",
    "emperor.layers._layer.Layer": "Layer",
    "emperor.layers._config.LayerConfig": "LayerConfig",
    "emperor.layers._stack.LayerStack": "LayerStack",
    "emperor.layers._config.LayerStackConfig": "LayerStackConfig",
    "emperor.layers._composition.recurrent.variants.standard.RecurrentLayer": "RecurrentLayer",
    "emperor.layers._composition.recurrent.config.RecurrentLayerConfig": "RecurrentLayerConfig",
    "emperor.layers._composition.recurrent.config.RecurrentCompositionConfig": "RecurrentCompositionConfig",
    "emperor.layers._composition.recurrent.variants.tiny_recursive_model.TinyRecursiveModelRecurrent": "TinyRecursiveModelRecurrent",
    "emperor.layers._composition.recurrent.config.TinyRecursiveModelRecurrentConfig": "TinyRecursiveModelRecurrentConfig",
    "emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.HierarchicalReasoningModelRecurrent": "HierarchicalReasoningModelRecurrent",
    "emperor.layers._composition.recurrent.config.HierarchicalReasoningModelRecurrentConfig": "HierarchicalReasoningModelRecurrentConfig",
    "emperor.experts._layers.mixture.MixtureOfExperts": "MixtureOfExperts",
    "emperor.experts._layers.map.MixtureOfExpertsMap": "MixtureOfExpertsMap",
    "emperor.experts._layers.reduce.MixtureOfExpertsReduce": "MixtureOfExpertsReduce",
    "emperor.experts._layers.layer.MixtureOfExpertsLayer": "MixtureOfExpertsLayer",
    "emperor.experts._config.MixtureOfExpertsConfig": "MixtureOfExpertsConfig",
    "emperor.experts._config.MixtureOfExpertsLayerConfig": "MixtureOfExpertsLayerConfig",
    "emperor.experts._config.MixtureOfExpertsModelConfig": "MixtureOfExpertsModelConfig",
    "emperor.layers._composition.gate.LayerGate": "LayerGate",
    "emperor.layers._config.GateConfig": "GateConfig",
    "emperor.layers._composition.residual.variants.additive.AdditiveResidual": "AdditiveResidual",
    "emperor.layers._composition.residual.variants.weighted.WeightedResidual": "WeightedResidual",
    "emperor.layers._composition.residual.variants.weighted_blend.WeightedBlendResidual": "WeightedBlendResidual",
    "emperor.layers._composition.residual.variants.attention.AttentionResidual": "AttentionResidual",
    "emperor.layers._composition.residual.config.ResidualConfig": "ResidualConfig",
    "emperor.layers._composition.residual.config.AdditiveResidualConfig": "AdditiveResidualConfig",
    "emperor.layers._composition.residual.config.WeightedResidualConfig": "WeightedResidualConfig",
    "emperor.layers._composition.residual.config.WeightedBlendResidualConfig": "WeightedBlendResidualConfig",
    "emperor.layers._composition.residual.config.AttentionResidualConfig": "AttentionResidualConfig",
    "emperor.halting._config.HaltingConfig": "HaltingConfig",
    "emperor.halting._variants.soft.SoftHalting": "SoftHalting",
    "emperor.halting._config.SoftHaltingConfig": "SoftHaltingConfig",
    "emperor.halting._variants.stick_breaking.StickBreaking": "StickBreaking",
    "emperor.halting._config.StickBreakingConfig": "StickBreakingConfig",
    "emperor.neuron._cluster.model.NeuronCluster": "NeuronCluster",
    "emperor.neuron._config.NeuronClusterConfig": "NeuronClusterConfig",
}

PROJECT_MODEL_DESCRIPTION: str = _DESCRIPTION_TEXT_BY_POLICY["Model"]
DESCRIPTION_BY_TYPE_ID: dict[str, str] = {
    type_id: _DESCRIPTION_TEXT_BY_POLICY[policy]
    for type_id, policy in DESCRIPTION_POLICY_BY_TYPE_ID.items()
}

INTERNAL_ROLE_TYPE_IDS: frozenset[str] = frozenset(
    {
        "torch.nn.modules.dropout.Dropout",
        "emperor.attention._ops.bias.KeyValueBias",
        "torch.nn.modules.normalization.LayerNorm",
        "emperor.layers._composition.recurrent.runtime.iteration_schedule.RecurrentIterationSchedule",
        "emperor.sampler._selection.losses.SamplerAuxiliaryLosses",
        "emperor.attention._variants.self_attention.processing.SelfAttentionProcessor",
        "emperor.attention._variants.self_attention.projection.SelfAttentionProjector",
        "torch.nn.modules.fold.Unfold",
    }
)
RUNTIME_ROLE_TYPE_IDS: frozenset[str] = frozenset(
    {
        "emperor.experiments.classifier._metrics.ClassifierMetricsLogger",
        "torch.nn.modules.loss.CrossEntropyLoss",
        "emperor.experiments.language_model._metrics.LanguageModelMetricsLogger",
        "torchmetrics.classification.accuracy.MulticlassAccuracy",
        "torchmetrics.classification.f_beta.MulticlassF1Score",
        "emperor.experiments.sequence_classifier._metrics.SequenceClassifierMetricsLogger",
    }
)

DEFAULT_RESIDUAL_OPTION_DESCRIPTION = (
    "Residual connection behavior. Enabled options require input_dim == output_dim."
)
DEFAULT_RESIDUAL_MODEL_DESCRIPTION = (
    "Optional model that generates data-dependent coefficients for weighted residual "
    "modes. When omitted, weighted modes use a learned scalar parameter."
)
RESIDUAL_FIELD_DESCRIPTIONS_BY_CONFIG_TYPE_ID: dict[str, tuple[str, str]] = {
    "emperor.layers._composition.recurrent.config.RecurrentLayerConfig": (
        "Residual connection behavior between recurrent steps. Set to null to "
        "disable recurrent residuals.",
        DEFAULT_RESIDUAL_MODEL_DESCRIPTION,
    ),
    "emperor.transformer._config.TransformerEncoderLayerConfig": (
        "Residual connection behavior applied to every encoder sub-block join.",
        "Optional data-dependent coefficient model used at each encoder join.",
    ),
    "emperor.transformer._config.TransformerDecoderLayerConfig": (
        "Residual connection behavior applied to every decoder sub-block join.",
        "Optional data-dependent coefficient model used at each decoder join.",
    ),
}

__all__ = [
    "DEFAULT_RESIDUAL_MODEL_DESCRIPTION",
    "DEFAULT_RESIDUAL_OPTION_DESCRIPTION",
    "DESCRIPTION_BY_TYPE_ID",
    "INTERNAL_ROLE_TYPE_IDS",
    "PROJECT_MODEL_DESCRIPTION",
    "RESIDUAL_FIELD_DESCRIPTIONS_BY_CONFIG_TYPE_ID",
    "RUNTIME_ROLE_TYPE_IDS",
]
