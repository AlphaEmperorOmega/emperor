from enum import Enum
import torch.nn as nn
from Emperor.components.parameter_generators.utils.linears import (
    DynamicDiagonalLinearLayer,
    LinearLayer,
)

from Emperor.components.parameter_generators.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    ParameterLayerBase,
    VectorParameterLayer,
)

from Emperor.experiments.layers.layers_models import (
    MultiLayerClassifierModel,
    SingleLayerClassifierModel,
)
from Emperor.experiments.layers.layers_presets import (
    FashionMNISTModelTrainer,
    FullMixtureLayerDynamicMaskPreset,
    FullMixtureNoWeightSumDepthFivePreset,
    FullMixtureNoWeightSumDepthFourPreset,
    FullMixtureNoWeightSumDepthThreePreset,
    FullMixtureNoWeightSumDepthTwoPreset,
    FullMixturePreset,
    FullMixtureThresholdPreset,
    SparseAuxiliaryAndWeightedParametersPreset,
    SparseNoAuxiliaryPreset,
    SparseThresholdPreset,
    SparseWithAuxiliaryPreset,
    TopKAuxiliaryAndWeightedParametersPreset,
    TopKNoAuxiliaryPreset,
    TopKThresholdPreset,
    TopkWithAuxiliaryPreset,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.experiments.layers.layers_models import ClassifierExperiment


class ModelFactory(Enum):
    SINGLE_LAYER = SingleLayerClassifierModel
    MULTI_LAYER = MultiLayerClassifierModel

    def create(
        self,
        model: "ParameterLayerBase",
        learning_rate: float = 0.1,
    ) -> "ClassifierExperiment":
        return self.value(model, learning_rate)


class PresetCollections:
    def __init__(self):
        pass

    def get_layers_and_assigned_presets(self):
        return {
            LinearLayer: self.__get_linear_presets(),
            DynamicDiagonalLinearLayer: self.__get_dynamic_diagonal_presets(),
            VectorParameterLayer: self.__get_vector_presets(),
            MatrixParameterLayer: self.__get_matrix_presets(),
            GeneratorParameterLayer: self.__get_generator_presets(),
        }

    def __get_linear_presets(self) -> list:
        return [SparseWithAuxiliaryPreset]

    def __get_dynamic_diagonal_presets(self) -> list:
        return [SparseWithAuxiliaryPreset]

    def __get_vector_presets(self) -> list:
        return [
            SparseWithAuxiliaryPreset,
            TopkWithAuxiliaryPreset,
            SparseNoAuxiliaryPreset,
            TopKNoAuxiliaryPreset,
            SparseAuxiliaryAndWeightedParametersPreset,
            TopKAuxiliaryAndWeightedParametersPreset,
            FullMixturePreset,
            FullMixtureLayerDynamicMaskPreset,
            SparseThresholdPreset,
            TopKThresholdPreset,
            FullMixtureThresholdPreset,
        ]

    def __get_matrix_presets(self) -> list:
        return [
            SparseWithAuxiliaryPreset,
            TopkWithAuxiliaryPreset,
            SparseNoAuxiliaryPreset,
            TopKNoAuxiliaryPreset,
            SparseAuxiliaryAndWeightedParametersPreset,
            TopKAuxiliaryAndWeightedParametersPreset,
            FullMixturePreset,
            FullMixtureLayerDynamicMaskPreset,
            SparseThresholdPreset,
            TopKThresholdPreset,
            FullMixtureThresholdPreset,
        ]

    def __get_generator_presets(self) -> list:
        return [
            SparseWithAuxiliaryPreset,
            TopkWithAuxiliaryPreset,
            SparseNoAuxiliaryPreset,
            TopKNoAuxiliaryPreset,
            SparseAuxiliaryAndWeightedParametersPreset,
            TopKAuxiliaryAndWeightedParametersPreset,
            FullMixturePreset,
            FullMixtureLayerDynamicMaskPreset,
            SparseThresholdPreset,
            TopKThresholdPreset,
            FullMixtureThresholdPreset,
            FullMixtureNoWeightSumDepthTwoPreset,
            FullMixtureNoWeightSumDepthThreePreset,
            FullMixtureNoWeightSumDepthFourPreset,
            FullMixtureNoWeightSumDepthFivePreset,
        ]


class ModelTrainer:
    def __init__(
        self,
        layer_preset,
        model_type,
        layer_type,
        learning_rate,
        trainer_type,
        mini_datasetset_flag,
    ):
        self.cfg = layer_preset.create().get_preset_config()
        self.model = model_type(self.cfg, layer_type, learning_rate)
        self.trainer = trainer_type(
            self.model,
            self.cfg,
            mini_datasetset_flag,
        )

    def train(self):
        self.trainer.train()


class TrainPresetsWrapper:
    def __init__(self) -> None:
        self.preset_collections = PresetCollections().get_layers_and_assigned_presets()
        self.learning_rates = [
            # 1e-5,
            # 3e-5,
            # 5e-5,
            # 7e-5,
            # 1e-4,
            # 3e-4,
            # 5e-4,
            # 7e-4,
            1e-3,
            # 3e-3,
            1e-2,
            # 3e-2,
            1e-1,
        ]

        self.model_types = [
            SingleLayerClassifierModel,
        ]
        self.trainer_types = [
            FashionMNISTModelTrainer,
        ]

    def test_all_preset_models(
        self,
        mini_datasetset_flag: bool = True,
    ) -> None:
        for learning_rate in self.learning_rates:
            for layer_type, all_layer_type_presets in self.preset_collections.items():
                for trainer_type in self.trainer_types:
                    for model_type in self.model_types:
                        for layer_preset in all_layer_type_presets:
                            self.print_model_title(
                                layer_type, layer_preset, learning_rate
                            )
                            trainer = ModelTrainer(
                                layer_preset,
                                model_type,
                                layer_type,
                                learning_rate,
                                trainer_type,
                                mini_datasetset_flag,
                            )
                            trainer.train()

    def print_model_title(
        self,
        layer_type,
        layer_preset,
        learning_rate: float,
    ) -> None:
        print("\n" * 2 + "-" * 50)
        model_type_msg = f"Model type: {layer_type.__name__} "
        model_preset_msg = f" Model preset: {layer_preset.__name__} "
        learning_rate_msg = f" Learning rate: {learning_rate} "
        message = "\n " + model_type_msg + model_preset_msg + learning_rate_msg + " \n"
        print(message)
