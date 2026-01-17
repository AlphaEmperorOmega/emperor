from Emperor.experiments.models.bert_vit_transformer import (
    BERTVITExperiment,
    BERTVITExperimentOptions,
)
from Emperor.linears.experiments import LinearsExperiments
from Emperor.transformer.experiments import TransformerExperiments
from Emperor.experts.experiments import MixtureOfExpertsExperiments
from Emperor.adaptive.experiments import AdaptiveParameterExperiments


if __name__ == "__main__":
    options = {
        "mini_datasetset_flag": False,
    }
    # LinearsExperiments(**options).train_base_model()
    # LinearsExperiments(**options).train_dynamic_model()
    # LinearsExperiments(**options).test_all_linear_types()
    # MixtureOfExpertsExperiments(**options).test_all_types()
    # AdaptiveParameterExperiments(**options).test_all_types()
    # TransformerExperiments(**options).train_transformer_BERT_VIT_model()

    BERTVITExperiment(**options).train_model(BERTVITExperimentOptions.ADAPTIVE_VIT)
