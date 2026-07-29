from emperor.datasets.text.question_answering._adapter import (
    _QuestionAnsweringAdapter,
    _SQuADv2Dataset,
)


class SQuADv2(_QuestionAnsweringAdapter):
    vocab_size: int = 97854  # approximate SQuAD v2 vocab size
    _source_name = "squad_v2"
    _item_dataset_type = _SQuADv2Dataset
