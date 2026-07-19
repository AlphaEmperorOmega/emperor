from emperor.datasets.text.language_modeling import PennTreebank, WikiText2
from emperor.experiments.tasks import ExperimentTask

DEFAULT_EXPERIMENT_TASK: ExperimentTask = ExperimentTask.CAUSAL_LANGUAGE_MODELING
DATASET_OPTIONS_BY_TASK: dict[ExperimentTask, list[type]] = {
    DEFAULT_EXPERIMENT_TASK: [WikiText2, PennTreebank],
}
