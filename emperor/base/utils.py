import copy
import random
import torch
import torch.nn as nn
import inspect
import collections
import IPython.display as display
import matplotlib.pyplot as plt

from lightning import LightningDataModule, LightningModule

from typing_extensions import Dict
from dataclasses import dataclass, fields, field, asdict
from torch.nn import Parameter, Linear, Sequential

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from emperor.config import ModelConfig

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
# device = "cpu"

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


class Module(LightningModule):
    """The base class of models."""

    def __init__(
        self, plot_train_per_epoch=2, plot_valid_per_epoch=1, plotProgress=True
    ):
        super().__init__()
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch
        self.plotProgress = plotProgress
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

    def test_step(self, batch):
        modelOutput, auxilaryLoss = self(*batch[:-1])
        loss = self.loss(modelOutput, batch[-1])
        if auxilaryLoss is not None:
            loss += auxilaryLoss

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

    def _override_config(
        self,
        cfg: "ConfigBase | ModelConfig",
        overrides: "ConfigBase | None" = None,
    ) -> "ConfigBase":
        if overrides is None:
            return cfg

        cfg = copy.deepcopy(cfg)
        for value in cfg.__dataclass_fields__:
            if (
                hasattr(overrides, "__dataclass_fields__")
                and value in overrides.__dataclass_fields__
            ):
                if getattr(overrides, value) is not None:
                    setattr(cfg, value, getattr(overrides, value))

        return cfg

    def _validate_fields(self, config: "ConfigBase", config_type: "ConfigBase") -> None:
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

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> None:
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

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

    def construct(
        self,
        class_type: type | None,
        cfg: "ConfigBase | None" = None,
    ) -> object | None:
        if class_type is None:
            return None
        if cfg is None:
            return class_type()
        return class_type(cfg)


class ParameterBank(Module):
    def __init__(
        self,
        shape: tuple,
        initializer: callable,
    ):
        super().__init__()
        self.shape = shape
        self.initializer = initializer
        self.parameter_bank = self.__create_bank()

    def __create_bank(self) -> Parameter:
        default_params = randn(*self.shape)
        parameter_bank = Parameter(default_params)
        self.initializer(parameter_bank)
        return parameter_bank

    def get(self) -> Parameter:
        return self.parameter_bank


class DataModule(LightningDataModule):
    """The base class of data."""

    def __init__(
        self,
        root="data",
        num_workers=4,
    ):
        super().__init__()
        self.root = root
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self._setup_fit()
            case "validate":
                self._setup_validate()
            case "test":
                self._setup_test()

    def _setup_fit(self) -> None:
        raise NotImplementedError(
            "The method '_setup_fit' must be implemented in the subclass."
        )

    def _setup_validate(self) -> None:
        raise NotImplementedError(
            "The method '_setup_validate' must be implemented in the subclass."
        )

    def _setup_test(self) -> None:
        raise NotImplementedError(
            "The method '_setup_test' must be implemented in the subclass."
        )

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def test_dataloader(self):
        return self._get_test_dataloader()

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self._text_labels(y)
        show_images(X.squeeze(1).permute(0, 2, 3, 1), nrows, ncols, titles=labels)

    def _text_labels(self, indices) -> list:
        raise NotImplementedError(
            "The 'test_labels' method must be implemented in the subclass."
        )


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

    def fit(
        self,
        model,
        data,
        print_loss_flag: bool = False,
        print_loss_frequency: int = 50,
    ):
        self.print_loss_flag = print_loss_flag
        self.print_loss_frequency = print_loss_frequency
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss, auxiliary_loss = self.model.training_step(self.prepare_batch(batch))
            self.__print_batch_messages(loss, auxiliary_loss, batch_idx=batch_idx)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch_idx, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                loss, auxiliary_loss = self.model.validation_step(
                    self.prepare_batch(batch)
                )
                self.__print_batch_messages(loss, auxiliary_loss, batch_idx=batch_idx)

    def __print_batch_messages(
        self,
        loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        batch_idx: int = 0,
    ) -> None:
        if self.print_loss_flag and batch_idx % self.print_loss_frequency == 0:
            message = [
                f"Epoch: {self.epoch}",
                f"Batch: {batch_idx}",
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
class ConfigBase:
    override_config: "ConfigBase | None" = field(
        default=None,
        metadata={"help": ""},
    )

    def build(self, overrides: "ConfigBase | None" = None) -> "Module":
        raise NotImplementedError

    def get(self, key: str, default=None) -> Any:
        if not hasattr(self, key):
            return None

        return getattr(self, key, default)

    def __post_init__(self):
        self._passed_args: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "passed_args":
                continue
            value = getattr(self, f.name)
            if (f.default is not None and value != f.default) or isinstance(
                value, bool
            ):
                self._passed_args[f.name] = value

    def get_custom_parameters(self) -> Dict[str, Any]:
        return self._passed_args

    def update(self, other: "ConfigBase") -> "ConfigBase":
        other_dict = asdict(other)
        for key, value in other_dict.items():
            if value is not None:
                setattr(self, key, value)
        return self


class ConfigUtils:
    @staticmethod
    def get_method_arguments():
        frame = inspect.currentframe()
        parent_frame = frame.f_back
        parent_arguments = inspect.getargvalues(parent_frame)
        parent_inputs = {
            key: parent_arguments.locals[key]
            for key in parent_arguments.locals
            if key in parent_arguments.args
        }

        del frame
        del parent_frame
        return parent_inputs
