from emperor.datasets.text.language_modeling import (
    OpenWebText,
    PennTreebank,
    WikiText2,
    WikiText103,
)
from emperor.experiments import ExperimentTask

DEFAULT_EXPERIMENT_TASK: ExperimentTask = ExperimentTask.CAUSAL_LANGUAGE_MODELING
DATASET_OPTIONS_BY_TASK: dict[ExperimentTask, list[type]] = {
    DEFAULT_EXPERIMENT_TASK: [WikiText2, PennTreebank, WikiText103, OpenWebText],
}
