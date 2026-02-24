from lightning import Trainer
from Emperor.datasets.image.cifar_10 import Cifar10
from models.linear import Model, ExperimentPresets, ExperimentOptions
from Emperor.datasets.image.mnist import Mnist


def train(
    dataset_type=Cifar10,
    config_option=ExperimentOptions.BASE,
    max_epochs=10,
    config_index=0,
):
    config = ExperimentPresets().get_config(config_option, dataset_type)[config_index]
    model = Model(cfg=config)
    data = dataset_type(batch_size=config.batch_size)

    trainer = Trainer(max_epochs=max_epochs, accelerator="cpu")
    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    return trainer, model


if __name__ == "__main__":
    train()
