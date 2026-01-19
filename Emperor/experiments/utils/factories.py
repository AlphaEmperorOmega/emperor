from Emperor.config import ModelConfig
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

        self.num_epochs = self._get_num_epochs()
        self.learning_rates = self._get_learning_rates()
        self.trainer_types = self._get_dataset_trainers()
        self.model_config = self._get_model_config()
        self.experiment_type = self._get_experiment_type()
        self.model_type = self._get_model_type()

    def _get_num_epochs(self) -> int:
        return 10

    def _get_learning_rates(self) -> list:
        return [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    def _get_dataset_trainers(self) -> list:
        return [FashionMNISTModelTrainer]

    def _get_model_config(self) -> ModelConfig:
        raise NotImplementedError(
            "The method '_get_model_config' must be implemented in the subclass."
        )

    def _set_model_config(self, model_config: "ModelConfig") -> None:
        self.model_config = model_config

    def _get_experiment_type(self):
        return ClassifierExperiment

    def _get_model_type(self) -> type:
        raise NotImplementedError(
            "The method '_get_model_type' must be implemented in the subclass."
        )

    def train_model(self) -> None:
        for trainer_type in self.trainer_types:
            for learning_rate in self.learning_rates:
                model = self.experiment_type(
                    self.model_config, self.model_type, learning_rate
                )
                trainer = trainer_type(
                    model,
                    self.model_config,
                    self.mini_datasetset_flag,
                    num_epochs=self.num_epochs,
                )

                parameter_count = self.__print_model_parameter_count(model)
                self.__print_model_title(trainer_type, learning_rate, parameter_count)
                trainer.train()

    def __print_model_parameter_count(self, model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def __print_model_title(
        self,
        dataset_trainer,
        learning_rate: float,
        parameter_count: int,
    ) -> None:
        print("\n" * 2 + "#" * 50)
        message_parts = [
            f"# Model type: {self.model_type.__name__} \n",
            f"# Trainer type: {dataset_trainer.__name__} \n",
            f"# Learning rate: {learning_rate} \n",
            f"# Experiment type: {self.experiment_type.__name__} \n",
            f"# Model Parameter count: {parameter_count} \n",
            f"# Config option: {self.model_config_option} \n",
        ]
        message = "\n" + "".join(message_parts) + "\n"
        print(message)
