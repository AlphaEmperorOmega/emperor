from emperor.datasets.text.bert_pretraining import (
    PennTreebankBertPretraining,
    WikiText2BertPretraining,
)
from emperor.experiments.tasks import ExperimentTask

DEFAULT_EXPERIMENT_TASK: ExperimentTask = ExperimentTask.BERT_PRETRAINING
DATASET_OPTIONS_BY_TASK: dict[ExperimentTask, list[type]] = {
    DEFAULT_EXPERIMENT_TASK: [PennTreebankBertPretraining, WikiText2BertPretraining],
}
