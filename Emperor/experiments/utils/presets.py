import os
import datetime
from Emperor.base.utils import Trainer
from Emperor.base.datasets import FashionMNIST


class FashionMNISTModelTrainer:
    def __init__(
        self,
        model,
        cfg,
        test_dataset_flag: bool = True,
        num_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data = self.__create_dataset(test_dataset_flag)
        self.model = model
        self.__initialize_monitor()

        if test_dataset_flag:
            assert cfg.batch_size < 64, (
                "The entire mini dataset contains 64 samplers, ensure that the `batch_size` is smaller than 64"
            )
        self.trainer = Trainer(max_epochs=num_epochs)

    def __create_dataset(self, test_dataset_flag) -> FashionMNIST:
        return FashionMNIST(
            batch_size=self.cfg.batch_size,
            testDatasetFalg=test_dataset_flag,
        )

    def train(self) -> None:
        self.trainer.fit(self.model, self.data, print_loss_flag=True)

    def __initialize_monitor(self) -> None:
        trainer_name = (
            self.__class__.__name__
            + "_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        dataset_name = self.data.__class__.__name__
        model_name = self.model.__class__.__name__
        log_dir = os.path.join(
            trainer_name, dataset_name, model_name, str(self.model.lr)
        )
        self.model.initialize_monitor(log_dir=log_dir)
