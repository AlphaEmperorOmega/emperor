from emperor.datasets.text.translation import Multi30kDeEn, Multi30kEnDe
from emperor.experiments.tasks import ExperimentTask

DEFAULT_EXPERIMENT_TASK = ExperimentTask.TEXT_TRANSLATION
DATASET_OPTIONS_BY_TASK = {DEFAULT_EXPERIMENT_TASK: [Multi30kDeEn, Multi30kEnDe]}
