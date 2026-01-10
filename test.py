from Emperor.adaptive.experiments import AdaptiveParameterExperiments
from Emperor.experts.experiments import MixtureOfExpertsExperiments
from Emperor.linears.experiments import LinearsExperiments


if __name__ == "__main__":
    options = {
        "mini_datasetset_flag": False,
    }
    # LinearsExperiments(**options).train_base_model()
    # LinearsExperiments(**options).train_dynamic_model()
    # LinearsExperiments(**options).test_all_linear_types()
    # MixtureOfExpertsExperiments(**options).test_all_types()
    AdaptiveParameterExperiments(**options).test_all_types()
