from emperor.base.options import BaseOptions


class ExperimentTask(BaseOptions):
    IMAGE_CLASSIFICATION = 1
    BERT_PRETRAINING = 2


def experiment_task_name(task: ExperimentTask) -> str:
    return type(task).cli_name(task.name)


def experiment_task_label(task: ExperimentTask) -> str:
    return task.name.replace("_", " ").title()


def resolve_experiment_task(
    value: str | ExperimentTask | None,
) -> ExperimentTask | None:
    if value is None:
        return None
    if isinstance(value, ExperimentTask):
        return value
    return ExperimentTask.get_member(value)
