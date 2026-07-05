"""Human-readable descriptions for model configuration fields."""

from __future__ import annotations

from typing import Any


EXPLICIT_FIELD_DESCRIPTIONS = {
    "BATCH_SIZE": (
        "Sets the number of examples processed in each training batch. Larger "
        "batches can improve throughput but use more memory and change optimizer "
        "dynamics."
    ),
    "NUM_EPOCHS": (
        "Sets the maximum number of full passes over the training dataset. "
        "Training can still stop earlier when trainer limits, early stopping, "
        "or max step settings are active."
    ),
    "LEARNING_RATE": (
        "Sets the optimizer step size for parameter updates. Lower values train "
        "more cautiously; higher values can converge faster but may destabilize "
        "training."
    ),
    "INPUT_DIM": (
        "Defines the flattened input feature dimension expected by the model. "
        "This should match the selected dataset or input projection."
    ),
    "OUTPUT_DIM": (
        "Defines the number of output logits produced by the model. For "
        "classification runs this should match the selected dataset class count."
    ),
    "DATASET_OPTIONS": (
        "Lists the datasets this model package can train against. The Viewer and "
        "CLI use these options to populate dataset selectors."
    ),
    "RUN_TEST_AFTER_FIT": (
        "Controls whether the trainer runs the test loop after fitting finishes. "
        "Disable this for quick smoke tests when you only need to verify training "
        "and validation."
    ),
    "TRAINER_ACCELERATOR": (
        "Selects the hardware backend used by PyTorch Lightning, such as CPU, "
        "GPU, or auto-detection. Use this when forcing a run onto specific "
        "hardware."
    ),
    "TRAINER_DEVICES": (
        "Selects how many devices or which device IDs the trainer should use. "
        "Keep this aligned with the chosen accelerator and available hardware."
    ),
    "TRAINER_GRADIENT_CLIP_VAL": (
        "Sets the gradient clipping threshold used before optimizer steps. Use a "
        "positive value to reduce exploding gradients in unstable runs."
    ),
    "TRAINER_GRADIENT_CLIP_ALGORITHM": (
        "Chooses how gradient clipping is applied, such as clipping by norm or "
        "by value. This only matters when the gradient clipping value is active."
    ),
    "TRAINER_ACCUMULATE_GRAD_BATCHES": (
        "Accumulates gradients across this many batches before each optimizer "
        "step. Values above 1 simulate a larger effective batch size but make "
        "max steps consume more batches."
    ),
    "TRAINER_PRECISION": (
        "Controls numeric precision for training, such as full precision or "
        "mixed precision. Changing this can affect memory use, speed, and "
        "numerical stability."
    ),
    "TRAINER_DETERMINISTIC": (
        "Requests deterministic operations from the trainer where possible. "
        "Enable this for reproducibility checks, with the tradeoff that some "
        "operations may run slower."
    ),
    "TRAINER_BENCHMARK": (
        "Allows backend benchmarking to choose faster kernels for fixed input "
        "shapes. Disable this when deterministic behavior is more important "
        "than throughput."
    ),
    "TRAINER_MAX_STEPS": (
        "Stops training after this many optimizer steps. Use a positive value "
        "for smoke tests or bounded experiments; -1 lets training run according "
        "to the epoch limit."
    ),
    "TRAINER_MAX_TIME": (
        "Stops training after a wall-clock time budget accepted by PyTorch "
        "Lightning. Leave it unset when training should be bounded by epochs or "
        "steps instead."
    ),
    "TRAINER_VAL_CHECK_INTERVAL": (
        "Controls how often validation runs during training. Use 1.0 for once "
        "per epoch, a fraction for within-epoch checks, or a batch interval when "
        "supported by the trainer."
    ),
    "TRAINER_LIMIT_TRAIN_BATCHES": (
        "Limits how much training data is used in each epoch. Use a fraction "
        "such as 0.1 for quick smoke tests, or an integer batch count when you "
        "need a fixed small run."
    ),
    "TRAINER_LIMIT_VAL_BATCHES": (
        "Limits how much validation data is evaluated each validation pass. Use "
        "a fraction such as 0.1 for 10% of validation batches during quick "
        "checks."
    ),
    "TRAINER_OVERFIT_BATCHES": (
        "Forces training and validation to reuse a tiny batch subset. Use this "
        "to check whether the model can overfit before running a full training "
        "job."
    ),
    "TRAINER_NUM_SANITY_VAL_STEPS": (
        "Runs this many validation batches before training starts. These sanity "
        "checks catch validation-loop errors before spending time on fitting."
    ),
    "TRAINER_LOG_EVERY_N_STEPS": (
        "Controls how often trainer metrics are logged in training steps. Lower "
        "values give more granular logs but increase logging overhead."
    ),
    "TRAINER_ENABLE_PROGRESS_BAR": (
        "Controls whether PyTorch Lightning renders a progress bar. Disable it "
        "for quieter logs in automated or Viewer-launched runs."
    ),
    "TRAINER_ENABLE_CHECKPOINTING": (
        "Controls Lightning's built-in checkpointing behavior. This is separate "
        "from explicit Viewer or callback checkpoint settings."
    ),
    "TRAINER_ENABLE_MODEL_SUMMARY": (
        "Controls whether the trainer prints a model summary before training. "
        "Enable it when inspecting layer structure; disable it to keep logs "
        "compact."
    ),
    "TRAINER_PROFILER": (
        "Selects a PyTorch Lightning profiler to collect runtime performance "
        "details. Leave it unset for normal runs because profiling adds overhead."
    ),
    "DATA_NUM_WORKERS": (
        "Sets the number of worker processes used by supported data modules for "
        "loading batches. Increase it to improve input throughput, or use 0 for "
        "simpler debugging."
    ),
    "CALLBACK_EARLY_STOPPING_PATIENCE": (
        "Sets how many validation checks can pass without improvement before "
        "early stopping halts training. Use 0 to disable the early stopping "
        "callback."
    ),
    "CALLBACK_EARLY_STOPPING_METRIC": (
        "Selects the logged validation metric watched by early stopping and "
        "checkpoint callbacks. The metric name must be emitted during "
        "validation."
    ),
    "CALLBACK_EARLY_STOPPING_MIN_DELTA": (
        "Sets the minimum metric change that counts as an improvement for early "
        "stopping. Increase it to ignore tiny fluctuations."
    ),
    "CALLBACK_EARLY_STOPPING_STRICT": (
        "Controls whether missing early-stopping metrics raise an error. Keep it "
        "enabled when a missing metric should fail the run loudly."
    ),
    "CALLBACK_EARLY_STOPPING_CHECK_FINITE": (
        "Controls whether early stopping aborts when the monitored metric "
        "becomes NaN or infinite. Keep it enabled to fail unstable runs quickly."
    ),
    "CALLBACK_CHECKPOINT_FLAG": (
        "Controls whether a model checkpoint callback is added for the selected "
        "run. When enabled, checkpoint selection follows the configured early "
        "stopping metric."
    ),
    "STACK_APPLY_OUTPUT_PIPELINE_FLAG": (
        "Controls whether the main layer stack applies its configured output "
        "pipeline after each layer. Disable it only when testing raw layer "
        "outputs or specialized stack behavior."
    ),
    "HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG": (
        "Controls whether the dedicated halting stack applies its output "
        "pipeline. Only matters when halting uses an independent stack; None "
        "lets the builder inherit the default stack behavior."
    ),
    "HALTING_STACK_BIAS_FLAG": (
        "Controls whether linear layers in the dedicated halting stack include "
        "bias terms. Only matters when halting uses its own stack; None lets the "
        "builder inherit the default stack behavior."
    ),
}

SECTION_CONTEXT = {
    "Layer Stack Options": "main layer stack",
    "Layer Stack Submodule Options": "shared layer-stack submodules",
    "Adaptive Generator Stack Options": "adaptive-parameter generator submodules",
    "Gate Options": "layer gating controller",
    "Gate Stack Options": "dedicated gate stack",
    "Halting Options": "halting controller",
    "Halting Stack Options": "dedicated halting stack",
    "Memory Options": "dynamic memory controller",
    "Memory Stack Options": "dedicated memory stack",
    "Recurrent Layer Options": "recurrent wrapper",
    "Recurrent Gate Options": "recurrent gate controller",
    "Recurrent Gate Stack Options": "dedicated recurrent gate stack",
    "Recurrent Halting Options": "recurrent halting controller",
    "Recurrent Halting Stack Options": "dedicated recurrent halting stack",
    "Weight Generator Options": "dynamic weight generator",
    "Weight Generator Stack Options": "dedicated weight-generator stack",
    "Bias Generator Options": "dynamic bias generator",
    "Bias Generator Stack Options": "dedicated bias-generator stack",
    "Diagonal Generator Options": "dynamic diagonal generator",
    "Diagonal Generator Stack Options": "dedicated diagonal-generator stack",
    "Mask Options": "row or column mask controller",
    "Mask Stack Options": "dedicated mask-generator stack",
    "Input Boundary Projector Options": "input boundary projector",
    "Output Boundary Projector Options": "output boundary projector",
}

FLAG_DISABLED_PREFIXES = (
    "GATE_",
    "HALTING_",
    "MEMORY_",
    "RECURRENT_",
    "WEIGHT_",
    "BIAS_",
    "DIAGONAL_",
    "MASK_",
)

INHERITED_STACK_SECTIONS = {
    "Gate Stack Options",
    "Halting Stack Options",
    "Memory Stack Options",
    "Recurrent Gate Stack Options",
    "Recurrent Halting Stack Options",
    "Weight Generator Stack Options",
    "Bias Generator Stack Options",
    "Diagonal Generator Stack Options",
    "Mask Stack Options",
}


def config_field_description(
    key: str,
    *,
    section: str,
    kind: str,
    nullable: bool,
    default: Any,
) -> str:
    """Return practical help text for a public config field."""

    if key in EXPLICIT_FIELD_DESCRIPTIONS:
        return EXPLICIT_FIELD_DESCRIPTIONS[key]

    subject = _subject_for_key(key, section)
    context = SECTION_CONTEXT.get(section, _humanize_section(section))
    caveat = _applicability_caveat(key, section, nullable, default)

    if key.endswith("_INDEPENDENT_FLAG"):
        return _with_caveat(
            f"Controls whether the {context} uses its own stack settings instead "
            "of inheriting shared submodule settings.",
            caveat,
        )
    if key.endswith("_BIAS_FLAG"):
        return _with_caveat(
            f"Controls whether linear layers in the {context} include bias terms.",
            caveat,
        )
    if key.endswith("_FLAG"):
        if subject == context:
            return _with_caveat(
                f"Enables or disables the {context}.",
                caveat,
            )
        return _with_caveat(
            f"Enables or disables {subject} for the {context}.",
            caveat,
        )
    if key.endswith("_HIDDEN_DIM"):
        return _with_caveat(
            f"Sets the hidden feature width used by the {context}. Larger values "
            "increase capacity and memory use.",
            caveat,
        )
    if key.endswith("_NUM_LAYERS"):
        return _with_caveat(
            f"Sets how many layers are built for the {context}. More layers add "
            "capacity and compute cost.",
            caveat,
        )
    if key.endswith("_DROPOUT_PROBABILITY") or key.endswith("_DROPOUT"):
        return _with_caveat(
            f"Sets the dropout rate used by the {context}. Increase it for more "
            "regularization, or use 0 to disable dropout.",
            caveat,
        )
    if key.endswith("_LAYER_NORM_POSITION"):
        return _with_caveat(
            f"Chooses where layer normalization is applied in the {context}. This "
            "changes training stability and the ordering of stack operations.",
            caveat,
        )
    if key.endswith("_ACTIVATION"):
        return _with_caveat(
            f"Selects the nonlinearity used by the {context}. The activation "
            "affects gradient flow and model expressiveness.",
            caveat,
        )
    if key.endswith("_OPTION"):
        return _with_caveat(
            f"Selects which implementation or behavior the {context} uses.",
            caveat,
        )
    if key.endswith("_THRESHOLD"):
        return _with_caveat(
            f"Sets the decision threshold used by the {context}. Higher values "
            "make activation or stopping conditions stricter.",
            caveat,
        )
    if key.startswith("TRAINER_"):
        return (
            f"Passes the {subject} setting through to the PyTorch Lightning "
            "trainer. Adjust it to control runtime training behavior rather than "
            "model architecture."
        )
    if key.startswith("CALLBACK_"):
        return (
            f"Configures the {subject} callback setting used around training. "
            "These values affect monitoring, early stopping, and checkpoint "
            "behavior rather than the model forward pass."
        )
    if key.startswith("DATA_"):
        return (
            f"Configures the {subject} data-loading setting for supported data "
            "modules. These values affect input pipeline behavior, not the model "
            "architecture."
        )
    if kind in {"int", "float"}:
        return _with_caveat(
            f"Sets the numeric {subject} value for the {context}. Tune it when "
            "changing capacity, regularization, or runtime limits.",
            caveat,
        )
    if kind == "bool":
        return _with_caveat(
            f"Turns the {subject} behavior on or off for the {context}.",
            caveat,
        )
    return _with_caveat(
        f"Configures the {subject} setting for the {context}.",
        caveat,
    )


def _subject_for_key(key: str, section: str) -> str:
    section_words = set(_humanize_section(section).split())
    words = [
        word
        for word in key.lower().split("_")
        if word not in section_words and word not in {"flag", "option"}
    ]
    if not words:
        return SECTION_CONTEXT.get(section, _humanize_section(section))
    return " ".join(words)


def _humanize_section(section: str) -> str:
    return section.strip().lower() if section.strip() else "this config section"


def _applicability_caveat(
    key: str,
    section: str,
    nullable: bool,
    default: Any,
) -> str:
    if key.endswith("_INDEPENDENT_FLAG"):
        return ""
    if (
        section in INHERITED_STACK_SECTIONS
        and not key.endswith("_INDEPENDENT_FLAG")
    ):
        return "Only matters when this stack is configured independently."
    if (
        any(key.startswith(prefix) for prefix in FLAG_DISABLED_PREFIXES)
        and not key.endswith("_FLAG")
        and not key.endswith("_INDEPENDENT_FLAG")
    ):
        root = key.split("_", 1)[0].lower()
        return f"Only applies when the {root} feature is enabled."
    if nullable and default is None:
        return "Use None to inherit the builder default when inheritance is supported."
    return ""


def _with_caveat(description: str, caveat: str) -> str:
    if not caveat:
        return description
    return f"{description} {caveat}"
