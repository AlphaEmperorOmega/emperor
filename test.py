# from Emperor.experiments.layers.layers_factories import TrainPresetsWrapper
#
# if __name__ == "__main__":
#     TrainPresetsWrapper().test_all_preset_models(False)

from Emperor.experiments.linears import LinearsExperiments


if __name__ == "__main__":
    options = {
        "mini_datasetset_flag": False,
    }
    LinearsExperiments(**options).train_base_model()
