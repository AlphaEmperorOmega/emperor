from __future__ import annotations

import sys
from dataclasses import dataclass
from types import MappingProxyType

from model_runtime.inspection.records import GraphRole


@dataclass(frozen=True, slots=True)
class _LazyTypeReference:
    module_name: str
    qualified_name: str

    @classmethod
    def parse(cls, reference: str) -> _LazyTypeReference:
        module_name, separator, qualified_name = reference.rpartition(".")
        if not separator or not module_name or not qualified_name:
            raise ValueError(f"Invalid semantic type reference: {reference!r}.")
        return cls(module_name, qualified_name)

    @property
    def key(self) -> tuple[str, str]:
        return self.module_name, self.qualified_name


@dataclass(frozen=True, slots=True)
class SemanticTypePolicy:
    type_reference: _LazyTypeReference
    description: str | None = None
    graph_role: GraphRole = "architecture"
    residual_field_descriptions: tuple[str, str] | None = None


class SemanticTypeCatalog:
    """Resolve Inspection semantics by exact registered type without importing it."""

    def __init__(self, policies: tuple[SemanticTypePolicy, ...]) -> None:
        policies_by_reference: dict[tuple[str, str], SemanticTypePolicy] = {}
        for policy in policies:
            key = policy.type_reference.key
            if key in policies_by_reference:
                reference = ".".join(key)
                raise ValueError(f"Duplicate semantic type policy for '{reference}'.")
            policies_by_reference[key] = policy
        self._policies_by_reference = MappingProxyType(policies_by_reference)

    @staticmethod
    def is_registered_type(class_type: type[object]) -> bool:
        registered: object = sys.modules.get(class_type.__module__)
        for name in class_type.__qualname__.split("."):
            if registered is None or name == "<locals>":
                return False
            registered = getattr(registered, name, None)
        return registered is class_type

    def policy_for(self, class_type: type[object]) -> SemanticTypePolicy | None:
        if not self.is_registered_type(class_type):
            return None
        return self._policies_by_reference.get(
            (class_type.__module__, class_type.__qualname__)
        )


DEFAULT_RESIDUAL_OPTION_DESCRIPTION = (
    "Residual connection behavior. Enabled options require input_dim == output_dim."
)
DEFAULT_RESIDUAL_MODEL_DESCRIPTION = (
    "Optional model that generates data-dependent coefficients for weighted residual "
    "modes. When omitted, weighted modes use a learned scalar parameter."
)


def _semantic_type_policy(
    type_reference: str,
    description: str | None = None,
    *,
    graph_role: GraphRole = "architecture",
    residual_field_descriptions: tuple[str, str] | None = None,
) -> SemanticTypePolicy:
    return SemanticTypePolicy(
        type_reference=_LazyTypeReference.parse(type_reference),
        description=description,
        graph_role=graph_role,
        residual_field_descriptions=residual_field_descriptions,
    )


_RECURRENT_RESIDUAL_DESCRIPTIONS = (
    "Residual connection behavior between recurrent steps. Set to null to "
    "disable recurrent residuals.",
    DEFAULT_RESIDUAL_MODEL_DESCRIPTION,
)
_TRANSFORMER_RESIDUAL_DESCRIPTIONS = (
    "Residual connection behavior applied to every encoder sub-block join.",
    "Optional data-dependent coefficient model used at each encoder join.",
)
_TRANSFORMER_DECODER_RESIDUAL_DESCRIPTIONS = (
    "Residual connection behavior applied to every decoder sub-block join.",
    "Optional data-dependent coefficient model used at each decoder join.",
)

PROJECT_MODEL_DESCRIPTION: str = "Top-level inspected model wrapper that owns the architecture, loss, metrics, and runtime modules for the selected preset."
SEMANTIC_TYPE_CATALOG = SemanticTypeCatalog(
    (
        _semantic_type_policy(
            "torch.nn.modules.container.ModuleList",
            "Container that stores an ordered list of child modules; execution is defined by the parent module.",
        ),
        _semantic_type_policy(
            "torch.nn.modules.container.Sequential",
            "Container that applies child modules in order, passing each output to the next child.",
        ),
        _semantic_type_policy(
            "torch.nn.modules.dropout.Dropout",
            "Regularization module that randomly zeroes activations during training and is inactive during evaluation.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "torch.nn.modules.normalization.LayerNorm",
            "Normalizes features within each sample to stabilize hidden-state scale before or after a layer block.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.runtime.iteration_schedule."
            "RecurrentIterationSchedule",
            "Tracks forward-call progress, active recurrent depth, and the gradient-enabled transition suffix.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "torch.nn.modules.loss.CrossEntropyLoss",
            "Runtime loss module for multi-class classification targets.",
            graph_role="runtime",
        ),
        _semantic_type_policy(
            "emperor.experiments.classifier._metrics.ClassifierMetricsLogger",
            "Runtime module that groups classifier metrics for train, validation, and test reporting.",
            graph_role="runtime",
        ),
        _semantic_type_policy(
            "emperor.experiments.language_model._metrics.LanguageModelMetricsLogger",
            "Runtime module that groups language-model metrics for train, validation, and test reporting.",
            graph_role="runtime",
        ),
        _semantic_type_policy(
            "emperor.experiments.sequence_classifier._metrics."
            "SequenceClassifierMetricsLogger",
            "Runtime module that groups sequence-classifier metrics for train, validation, and test reporting.",
            graph_role="runtime",
        ),
        _semantic_type_policy(
            "torchmetrics.classification.accuracy.MulticlassAccuracy",
            "Runtime metric that reports the share of classified examples whose predicted class matches the target class.",
            graph_role="runtime",
        ),
        _semantic_type_policy(
            "torchmetrics.classification.f_beta.MulticlassF1Score",
            "Runtime metric that reports the harmonic mean of classifier precision and recall across classes.",
            graph_role="runtime",
        ),
        _semantic_type_policy(
            "emperor.attention._ops.bias.KeyValueBias",
            "Internal attention helper that adds learned key/value bias terms.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.sampler._selection.losses.SamplerAuxiliaryLosses",
            "Internal mixture-of-experts helper that tracks auxiliary routing losses.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.attention._variants.self_attention.processing."
            "SelfAttentionProcessor",
            "Internal attention helper that prepares attention inputs and masks.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.attention._variants.self_attention.projection."
            "SelfAttentionProjector",
            "Internal attention helper that projects hidden states into attention query, key, and value tensors.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "torch.nn.modules.fold.Unfold",
            "Internal tensor reshaping module that extracts sliding local blocks from an input tensor.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.linears._layer.LinearLayer",
            "Applies a learned linear projection with configured input/output dimensions and optional bias.",
        ),
        _semantic_type_policy(
            "emperor.linears._config.LinearLayerConfig",
            "Builds a learned linear projection with configured input/output dimensions and optional bias.",
        ),
        _semantic_type_policy(
            "emperor.augmentations.adaptive_parameters._linear_adapter."
            "AdaptiveLinearLayer",
            "Applies a learned linear projection that can optionally augment parameters from the current input.",
        ),
        _semantic_type_policy(
            "emperor.augmentations.adaptive_parameters._config."
            "AdaptiveLinearLayerConfig",
            "Builds a linear projection that can optionally augment parameters from the current input.",
        ),
        _semantic_type_policy(
            "emperor.layers._layer.core.Layer",
            "Applies one configured layer block with optional activation, residuals, normalization, gating, halting, and memory hooks.",
        ),
        _semantic_type_policy(
            "emperor.layers._layer.pipeline.postprocessing.LayerPostprocessingDelegate",
            "Internal Layer Pipeline Module that applies activation, gating, and dropout in order.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.layers._layer.pipeline.halting.LayerHaltingDelegate",
            "Internal Layer Pipeline Module that owns halting updates and finalization.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.layers._layer.pipeline.memory.LayerMemoryDelegate",
            "Internal Layer Pipeline Module that applies dynamic memory at its configured position.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.layers._layer.pipeline.residual.LayerResidualDelegate",
            "Internal Layer Pipeline Module that owns residual state and composition.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.layers._layer.pipeline.normalization.LayerNormalizationDelegate",
            "Internal Layer Pipeline Module that applies normalization at its configured position.",
            graph_role="internal",
        ),
        _semantic_type_policy(
            "emperor.layers._config.LayerConfig",
            "Builds a Layer block with optional activation, residuals, normalization, gating, halting, and memory hooks.",
        ),
        _semantic_type_policy(
            "emperor.layers._stack.LayerStack",
            "Runs an ordered stack of Layer blocks, with shared dimensions and optional shared gate, halting, or memory modules.",
        ),
        _semantic_type_policy(
            "emperor.layers._config.LayerStackConfig",
            "Builds an ordered stack of Layer blocks, with shared dimensions and optional shared gate, halting, or memory modules.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.variants.standard.RecurrentLayer",
            "Reuses a configured block for multiple recurrent steps, optionally adding recurrent gating, normalization, halting, or memory.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.config.RecurrentLayerConfig",
            "Builds a recurrent block that can run for multiple steps with optional gating, normalization, halting, or memory.",
            residual_field_descriptions=_RECURRENT_RESIDUAL_DESCRIPTIONS,
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.config.RecurrentCompositionConfig",
            "Abstract recurrent composition Interface; use a concrete recurrent config.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.variants.tiny_recursive_model."
            "TinyRecursiveModelRecurrent",
            "Reuses one transition block for Tiny Recursive Model latent and answer updates across a fixed answer-update schedule.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.config."
            "TinyRecursiveModelRecurrentConfig",
            "Builds Tiny Recursive Model recurrence with one shared transition, a latent-update count per answer update, and an answer-update count.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.variants."
            "hierarchical_reasoning_model.HierarchicalReasoningModelRecurrent",
            "Runs distinct low- and high-level transitions on nested recurrent clocks.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.recurrent.config."
            "HierarchicalReasoningModelRecurrentConfig",
            "Builds Hierarchical Reasoning Model recurrence with separate low- and high-level transitions and clock counts.",
        ),
        _semantic_type_policy(
            "emperor.experts._layers.mixture.MixtureOfExperts",
            "Routes inputs across a set of expert modules using sampler probabilities and combines or maps the selected expert outputs.",
        ),
        _semantic_type_policy(
            "emperor.experts._layers.map.MixtureOfExpertsMap",
            "Routes inputs across experts and returns mapped expert outputs.",
        ),
        _semantic_type_policy(
            "emperor.experts._layers.reduce.MixtureOfExpertsReduce",
            "Routes inputs across experts and reduces selected expert outputs back into one representation.",
        ),
        _semantic_type_policy(
            "emperor.experts._layers.layer.MixtureOfExpertsLayer",
            "Wraps mixture-of-experts routing in the standard Layer pipeline.",
        ),
        _semantic_type_policy(
            "emperor.experts._config.MixtureOfExpertsConfig",
            "Configures expert count, routing, capacity, weighting, sampler behavior, and expert model construction.",
        ),
        _semantic_type_policy(
            "emperor.experts._config.MixtureOfExpertsLayerConfig",
            "Builds a mixture-of-experts layer inside the standard Layer pipeline.",
        ),
        _semantic_type_policy(
            "emperor.experts._config.MixtureOfExpertsModelConfig",
            "Builds a model around a mixture-of-experts layer stack.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.gate.core.LayerGate",
            "Combines a learned gate output with the current layer value by scaling or addition.",
        ),
        _semantic_type_policy(
            "emperor.layers._config.GateConfig",
            "Configures a layer gate network and how its output is composed with the current value.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.variants.additive.AdditiveResidual",
            "Adds the current and previous hidden values.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.variants.weighted.WeightedResidual",
            "Adds the previous hidden value to a learned tanh-weighted current value.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.variants.weighted_blend."
            "WeightedBlendResidual",
            "Convexly blends current and previous hidden values with a learned weight.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.variants.attention.AttentionResidual",
            "Routes across forward-local residual-depth sources with learned attention.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.config.ResidualConfig",
            "Abstract residual configuration Interface; use a concrete residual config.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.config.AdditiveResidualConfig",
            "Builds direct additive residual composition.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.config.WeightedResidualConfig",
            "Builds tanh-weighted residual composition with a scalar or generated coefficient.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.config.WeightedBlendResidualConfig",
            "Builds sigmoid convex residual blending with a scalar or generated coefficient.",
        ),
        _semantic_type_policy(
            "emperor.layers._composition.residual.config.AttentionResidualConfig",
            "Builds learned attention routing across residual-depth history.",
        ),
        _semantic_type_policy(
            "emperor.halting._config.HaltingConfig",
            "Configures adaptive computation halting thresholds, dropout, hidden-state mode, and gate network.",
        ),
        _semantic_type_policy(
            "emperor.halting._variants.soft.SoftHalting",
            "Accumulates weighted recurrent states until the halting threshold is met.",
        ),
        _semantic_type_policy(
            "emperor.halting._config.SoftHaltingConfig",
            "Builds soft halting, which accumulates weighted recurrent states until the threshold is met.",
        ),
        _semantic_type_policy(
            "emperor.halting._variants.stick_breaking.StickBreaking",
            "Allocates remaining recurrent probability mass step by step until the halting threshold is met.",
        ),
        _semantic_type_policy(
            "emperor.halting._config.StickBreakingConfig",
            "Builds stick-breaking halting, which allocates remaining recurrent probability mass over steps.",
        ),
        _semantic_type_policy(
            "emperor.neuron._cluster.model.NeuronCluster",
            "Maintains a 3D cluster of routed neurons that can traverse, branch, and grow during training.",
        ),
        _semantic_type_policy(
            "emperor.neuron._config.NeuronClusterConfig",
            "Configures a 3D routed neuron cluster, including capacity, traversal, sampling, and growth controls.",
        ),
        _semantic_type_policy(
            "emperor.transformer._config.TransformerEncoderLayerConfig",
            residual_field_descriptions=_TRANSFORMER_RESIDUAL_DESCRIPTIONS,
        ),
        _semantic_type_policy(
            "emperor.transformer._config.TransformerDecoderLayerConfig",
            residual_field_descriptions=_TRANSFORMER_DECODER_RESIDUAL_DESCRIPTIONS,
        ),
    )
)

__all__ = [
    "DEFAULT_RESIDUAL_MODEL_DESCRIPTION",
    "DEFAULT_RESIDUAL_OPTION_DESCRIPTION",
    "PROJECT_MODEL_DESCRIPTION",
    "SEMANTIC_TYPE_CATALOG",
    "SemanticTypeCatalog",
    "SemanticTypePolicy",
]
