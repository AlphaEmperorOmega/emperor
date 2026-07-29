from emperor.datasets.text.question_answering._adapter import (
    _QuestionAnsweringAdapter,
    _SQuADv1Dataset,
)


class SQuADv1(_QuestionAnsweringAdapter):
    vocab_size: int = 87599  # approximate SQuAD v1 vocab size
    _source_name = "squad"
    _item_dataset_type = _SQuADv1Dataset
