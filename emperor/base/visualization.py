import collections
import inspect


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    import matplotlib.pyplot as plt

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs, strict=False)):
        try:
            img = img.detach().numpy()
        except Exception:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=None):
        """Save function arguments into class attribuites."""
        ignore = ignore or []
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
        ls=None,
        colors=None,
        fig=None,
        axes=None,
        figsize=(3.5, 2.5),
        display=True,
    ):
        ls = ls or ["-", "--", "-.", ":"]
        colors = colors or ["C0", "C1", "C2", "C3"]
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

        def mean(values):
            return sum(values) / len(values)

        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()

        if not self.display:
            return

        import IPython.display as display
        import matplotlib.pyplot as plt

        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)

        plt_lines, labels = [], []
        for (k, v), ls, color in zip(
            self.data.items(), self.ls, self.colors, strict=False
        ):
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


__all__ = ["HyperParameters", "ProgressBoard", "show_images"]
