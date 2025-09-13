from dataclasses import dataclass, field, fields
import torch
import torch.nn as nn
import inspect
import collections
import IPython.display as display
import matplotlib.pyplot as plt
import random
from torch.nn import Parameter, Linear, Sequential

from typing import TYPE_CHECKING, Any, Optional, Union


if TYPE_CHECKING:
    from Emperor.config import ModelConfig

ones_like = torch.ones_like
ones = torch.ones
zeros_like = torch.zeros_like
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
meshgrid = torch.meshgrid
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
randn = torch.randn
randn_like = torch.randn_like
matmul = torch.matmul
int32 = torch.int32
int64 = torch.int64
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
prod = torch.prod
masked_fill = torch.masked_fill
sigmoid = torch.sigmoid
batch_matmul = torch.bmm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)


def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    # backend_inline.set_matplotlib_formats('svg')


def cpu():
    """Get the CPU device."""
    return torch.device("cpu")


def gpu(i=0):
    """Get a GPU device."""
    return torch.device(f"cuda:{i}")


def num_gpus():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = numpy(img)
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attribuites."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        for k, v in self.hparams.items():
            setattr(self, k, v)


class ProgressBoard(HyperParameters):
    """The board that plots data points in animation."""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        ls=["-", "--", "-.", ":"],
        colors=["C0", "C1", "C2", "C3"],
        fig=None,
        axes=None,
        figsize=(3.5, 2.5),
        display=True,
    ):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple("Point", ["x", "y"])

        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()

        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []

        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))

        if len(points) != every_n:
            return

        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()

        if not self.display:
            return

        use_svg_display()

        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)

        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(
                plt.plot([p.x for p in v], [p.y for p in v], linestyle=ls, color=color)[
                    0
                ]
            )
            labels.append(k)

        axes = self.axes if self.axes else plt.gca()
        if self.xlim:
            axes.set_xlim(self.xlim)
        if self.ylim:
            axes.set_ylim(self.ylim)
        if not self.xlabel:
            self.xlabel = self.x

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)

        display.display(self.fig)
        display.clear_output(wait=True)


class Module(nn.Module, HyperParameters):
    """The base class of models."""

    def __init__(
        self, plot_train_per_epoch=2, plot_valid_per_epoch=1, plotProgress=True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        if self.plotProgress:
            assert hasattr(self, "trainer"), "Trainer is not inited"
            self.board.xlabel = "epoch"
            if train:
                x = self.trainer.train_batch_idx / self.trainer.num_train_batches
                n = self.trainer.num_train_batches / self.plot_train_per_epoch
            else:
                x = self.trainer.epoch + 1
                n = self.trainer.num_val_batches / self.plot_valid_per_epoch
            self.board.draw(
                x,
                numpy(to(value, cpu())),
                ("train_" if train else "val_") + key,
                every_n=int(n),
            )

    def training_step(self, batch):
        modelOutput, auxilary_loss = self(*batch[:-1])
        loss = self.loss(modelOutput, batch[-1])

        if auxilary_loss is not None:
            loss += auxilary_loss

        self.plot("loss", loss, train=True)
        return loss, auxilary_loss

    def validation_step(self, batch):
        modelOutput, auxilaryLoss = self(*batch[:-1])
        loss = self.loss(modelOutput, batch[-1])
        if auxilaryLoss is not None:
            loss += auxilaryLoss
        self.plot("loss", loss, train=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def _getValue(self, defaultValue, configValue):
        return defaultValue if defaultValue is not None else configValue

    def _resolve_config(self, cfg=None, cfg_key: str = ""):
        return getattr(cfg, cfg_key, cfg) if cfg_key and cfg else cfg

    def _resolveOld(
        self,
        defaultValue: Any,
        configKey: Any,
        cfg: Any = None,
    ) -> Any:
        if cfg is None and self.cfg is None:
            return defaultValue
        config = cfg if cfg is not None else self.cfg
        configValue = getattr(config, configKey, None)
        return defaultValue if defaultValue is not None else configValue

    def _resolve(
        self,
        configKey: Any,
        cfg: Any,
    ) -> Any:
        config = cfg if cfg is not None else self.cfg
        configValue = getattr(config, configKey, None)
        defaultValue = self.inputs.get(configKey)
        return defaultValue if defaultValue is not None else configValue

    def _get_config(self, cfg: Optional["Configuration"], configName: str) -> Any:
        return getattr(cfg, configName) if cfg is not None else None

    def _initialize_parameters(
        self, *parameters: Union[Linear, Parameter, Sequential]
    ) -> None:
        for parameter in parameters:
            if isinstance(parameter, Parameter):
                nn.init.xavier_uniform_(parameter)

            if isinstance(parameter, Linear):
                nn.init.xavier_uniform_(parameter.weight)
                if parameter.bias is not None:
                    nn.init.zeros_(parameter.bias)

            if isinstance(parameter, Sequential):
                for layer in parameter:
                    self._initialize_parameters(layer)

    def _overwrite_config(
        self,
        cfg: "DataClassBase | ModelConfig",
        overwrrides: "DataClassBase | None" = None,
    ) -> "DataClassBase":
        if overwrrides is None:
            return cfg
        for field in cfg.__dataclass_fields__:
            if hasattr(overwrrides, field) and getattr(overwrrides, field) is not None:
                setattr(cfg, field, getattr(overwrrides, field))
        return cfg

    def _validate_fields(
        self, config: "DataClassBase", config_type: "DataClassBase"
    ) -> None:
        for config_field in fields(config_type):
            field_value = getattr(config, config_field.name)
            is_field_value_none = field_value is None
            if is_field_value_none:
                if "required" in config_field.metadata:
                    if config_field.metadata["required"]:
                        raise ValueError(
                            f"{config_field.name} is required but it was not set."
                        )
                    return
                raise ValueError(f"{config_field.name} is required but it was not set.")

    def _init_parameter_bank(
        self,
        parameter_shape: tuple,
        initializer: callable = None,
    ) -> Parameter:
        # TODO: Ensure you have the option to initialize the biases with
        # as a zero zensor.
        initializer = (
            initializer if initializer is not None else self._initialize_parameters
        )
        bank = ParameterBank(parameter_shape, initializer)
        return bank.get()


class ParameterBank:
    def __init__(
        self,
        shape: tuple,
        initializer: callable,
    ):
        self.shape = shape
        self.initializer = initializer
        self.parameter_bank = self.__create_bank()

    def __create_bank(self) -> Parameter:
        parameter_bank = Parameter(randn(*self.shape))
        self.initializer(parameter_bank)
        return parameter_bank

    def get(self) -> Parameter:
        return self.parameter_bank


class DataModule(HyperParameters):
    """The base class of data."""

    def __init__(
        self,
        root="data",
        num_workers=4,
        testDatasetFalg=False,
        testDatasetNumSamples=512,
    ):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def getDataset(self, dataset):
        if self.testDatasetFalg:
            totalSamplesRange = range(len(dataset))
            small_dataset_idxs = random.sample(
                totalSamplesRange, self.testDatasetNumSamples
            )
            return torch.utils.data.Subset(dataset, small_dataset_idxs)
        return dataset


class Trainer(HyperParameters):
    def __init__(self, max_epochs, num_gpu=0, gradient_clip_val=0, view_progress=True):
        self.save_hyperparameters()
        self.print_loss = False
        self.gpus = [gpu(i) for i in range(min(num_gpu, num_gpus()))]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data, print_loss_flag: bool = False):
        self.print_loss_flag = print_loss_flag
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss, auxiliary_loss = self.model.training_step(self.prepare_batch(batch))
            self.__print_batch_messages(loss, auxiliary_loss)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        self.train_batch_idx = 0
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    def __print_batch_messages(
        self,
        loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        batch_rate: int = 1,
    ) -> None:
        if self.print_loss_flag and self.train_batch_idx % batch_rate == 0:
            message = [
                f"Epoch: {self.epoch}",
                f"Batch: {self.train_batch_idx}",
                f"Total loss: {round(loss.item(), 4)}",
                f"Model loss: {round(loss.item() - auxiliary_loss.item(), 4)}",
                f"Auxiliary loss: {round(auxiliary_loss.item(), 4)}",
            ]
            print(", ".join(message))

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [to(a, self.gpus[0]) for a in batch]
        return batch

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm


@dataclass
class DataClassBase:
    def get(self, key: str, default=None) -> Any:
        if not hasattr(self, key):
            return None

        return getattr(self, key, default)
