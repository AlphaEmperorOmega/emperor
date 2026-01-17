from Emperor.config import ModelConfig
from Emperor.base.enums import BaseOptions
from Emperor.experiments.utils.models import ClassifierExperiment
from Emperor.experiments.utils.presets import FashionMNISTModelTrainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.experiments.utils.models import ClassifierExperiment


class Experiments:
    def __init__(
        self,
        mini_datasetset_flag: bool = True,
        is_transformer_flag: bool = False,
    ) -> None:
        self.mini_datasetset_flag = mini_datasetset_flag
        self.is_transformer_flag = is_transformer_flag
        self.model_config = None
        self.learning_rates = [
            # 1e-5,
            # 1e-4,
            1e-3,
            # 1e-2,
            # 1e-1,
        ]
        self.trainer_types = [
            FashionMNISTModelTrainer,
        ]

    def _get_model_type(self):
        return ClassifierExperiment

    def _print_model_title(
        self,
        layer_type,
        layer_preset,
        learning_rate: float,
    ) -> None:
        print("\n" * 2 + "-" * 50)
        model_type_msg = ""  # f"Model type: {layer_type.__name__} "
        model_preset_msg = ""  # f" Model preset: {layer_preset.__name__} "
        learning_rate_msg = f" Learning rate: {learning_rate} "
        message = "\n " + model_type_msg + model_preset_msg + learning_rate_msg + " \n"
        print(message)

    def _train_model(
        self,
        layer_type: BaseOptions,
        print_parameter_count_flag: bool = False,
    ) -> None:
        layer_type = (
            layer_type.value if isinstance(layer_type, BaseOptions) else layer_type
        )
        for learning_rate in self.learning_rates:
            for trainer_type in self.trainer_types:
                self._print_model_title(
                    layer_type, self._get_model_config(), learning_rate
                )
                trainer = ModelTrainer(
                    self._get_model_config(),
                    self._get_model_type(),
                    layer_type,
                    learning_rate,
                    trainer_type,
                    self.mini_datasetset_flag,
                    num_epochs=20,
                )
                if print_parameter_count_flag:
                    trainer.print_model_parameter_count()
                trainer.train()

    def _get_model_config(self) -> None:
        if self.model_config is None:
            raise ValueError(
                "self.model_config is None. It must be set before calling this method."
            )
        return self.model_config

    def _set_model_config(self, model_config: "ModelConfig") -> None:
        self.model_config = model_config

    def test_all_preset_models(self) -> None:
        for learning_rate in self.learning_rates:
            for layer_type, all_layer_type_presets in self.preset_collections.items():
                for trainer_type in self.trainer_types:
                    for layer_preset in all_layer_type_presets:
                        self._print_model_title(layer_type, layer_preset, learning_rate)
                        trainer = ModelTrainer(
                            layer_preset,
                            self._get_model_type(),
                            layer_type,
                            learning_rate,
                            trainer_type,
                            self.mini_datasetset_flag,
                        )
                        trainer.train()


class ModelTrainer:
    def __init__(
        self,
        model_config,
        model_type,
        layer_type,
        learning_rate,
        trainer_type,
        mini_datasetset_flag,
        num_epochs: int = 5,
    ):
        self.cfg = model_config
        self.model = model_type(self.cfg, layer_type, learning_rate)

        self.trainer = trainer_type(
            self.model,
            self.cfg,
            mini_datasetset_flag,
            num_epochs=num_epochs,
        )

    def print_model_parameter_count(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameter count: {count}")

    def train(self):
        self.trainer.train()
