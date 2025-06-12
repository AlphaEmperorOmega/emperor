from enum import Enum

from Emperor.components.parameter_generators.layers import (
    DefaultLinearLayer,
    GeneratorParameterLayer,
    MatrixParameterLayer,
    ParameterLayerBase,
    VectorParameterLayer,
)

from typing import TYPE_CHECKING

from Emperor.experiments.layers.layers_models import (
    MultiLayerClassifierModel,
    SingleLayerClassifierModel,
)
from Emperor.experiments.layers.layers_presets import (
    FashionMNISTModelTrainer,
    ParameterLayerPresetFactory,
)

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
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


class ParameterLayerOptions(Enum):
    # DEFAULT = DefaultLinearLayer
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer

    def create(
        self,
        cfg: "ModelConfig",
    ) -> ParameterLayerBase:
        return self.value(cfg)


class ParameterLayerPresetOptions(Enum):
    SPARSE = "create_sparse_layer"
    TOPK = "create_topk_layer"
    FULL_MIXTURE = "create_full_mixture_layer"
    RANDOM_TOPK = "create_random_topk_layer"
    SPARSE_THRESHOLD = "create_sparse_threshold_layer"
    TOPK_THRESHOLD = "create_topk_threshold_layer"
    FULL_MIXTURE_THRESHOLD = "create_full_mixture_threshold_layer"
    SPARSE_NOISY_TOPK = "create_sparse_noisy_topk_layer"
    TOPK_NOISY_TOPK = "create_topk_noisy_topk_layer"
    FULL_MIXTURE_NOISY_TOPK = "create_full_mixture_noisy_topk_layer"

    def train_single_layer_fashion_minist_dataset(
        self,
        layer_type: "ParameterLayerOptions",
        mini_training_set_flag: bool = True,
    ) -> None:
        cfg = FashionMNISTModelTrainer.config_preset()
        parameter_layer_preset = ParameterLayerPresetFactory(layer_type, cfg)
        parameter_layer_preset = getattr(parameter_layer_preset, self.value)()
        model = SingleLayerClassifierModel(
            parameter_layer_preset,
            learning_rate=0.0001,
        )
        dataset = FashionMNISTModelTrainer(
            model,
            cfg,
            test_dataset_flag=mini_training_set_flag,
        )
        dataset.train()

    def train_multi_layer_fashion_minist_dataset(
        self,
        layer_preset: "ParameterLayerOptions",
        mini_training_set_flag: bool = True,
    ) -> None:
        cfg = FashionMNISTModelTrainer.config_preset()

        def create_hidden_layer_model(
            cfg: "ModelConfig",
            layer_shape: list[int],
            layer_type: "ParameterLayerBase | None" = None,
        ):
            input_dim, output_dim = layer_shape
            parameter_layer_preset = ParameterLayerPresetFactory(layer_preset, cfg)
            parameter_layer_preset.set_layer_input_dim(input_dim)
            parameter_layer_preset.set_layer_output_dim(output_dim)

            layer_type_method = self.value if layer_type is None else layer_type

            return getattr(parameter_layer_preset, self.value)()

        model = MultiLayerClassifierModel(
            cfg=cfg,
            hidden_layer_callback=create_hidden_layer_model,
            num_hidden_layers=2,
        )
        dataset = FashionMNISTModelTrainer(
            model,
            cfg,
            test_dataset_flag=mini_training_set_flag,
        )
        dataset.train()


class ParameterLayersPresetTester:
    def test_all_presets_single_layer_models(self):
        self.__test_all_preset_models(
            "train_single_layer_fashion_minist_dataset",
            mini_training_set_flag=True,
        )

    def test_all_presets_multi_layer_models(self):
        self.__test_all_preset_models(
            "train_multi_layer_fashion_minist_dataset",
            mini_training_set_flag=True,
        )

    def __test_all_preset_models(
        self, train_method_name: str, mini_training_set_flag: bool = True
    ) -> None:
        for layer_model in ParameterLayerOptions:
            for layer_preset in ParameterLayerPresetOptions:
                print("-" * 50)
                print(
                    f"\n Model type: {str(layer_model.name)} - preset: {str(layer_preset.name)}: \n"
                )
                getattr(layer_preset, train_method_name)(
                    layer_model,
                    mini_training_set_flag=mini_training_set_flag,
                )
                print("-" * 50 + "\n")
