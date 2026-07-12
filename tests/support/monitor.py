import torch


def same_bound_method(left, right):
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", left) is getattr(right, "__func__", right)
    )


class CaptureExperiment:
    def __init__(self):
        self.histograms = []
        self.images = []

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values.detach().clone(), step))

    def add_image(self, tag, image, step, dataformats=None):
        self.images.append((tag, image.detach().clone(), step, dataformats))


class CaptureLogger:
    def __init__(self):
        self.experiment = CaptureExperiment()


class CaptureLightningModule(torch.nn.Module):
    def __init__(self, **modules):
        super().__init__()
        self.global_step = 0
        self.logger = CaptureLogger()
        self.logged = []
        for name, module in modules.items():
            self.add_module(name, module)

    def log(self, tag, value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu()
        self.logged.append((tag, value))

    @property
    def logged_tags(self):
        return [tag for tag, _ in self.logged]

    def logged_value(self, tag):
        for logged_tag, value in reversed(self.logged):
            if logged_tag == tag:
                return value
        raise KeyError(tag)


class NoExperimentLightningModule(CaptureLightningModule):
    def __init__(self, **modules):
        super().__init__(**modules)
        self.logger = None


class TrainerStub:
    def __init__(self, global_step=0):
        self.global_step = global_step
