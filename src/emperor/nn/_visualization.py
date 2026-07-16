import collections
import inspect


class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=None):
        """Save function arguments into class attribuites."""
        ignore = ignore or []
        frame = inspect.currentframe()
        while frame is not None:
            frame = frame.f_back
            if frame is not None and frame.f_locals.get("self") is self:
                break
        if frame is None:
            raise RuntimeError(
                "save_hyperparameters must be called from an instance method."
            )
        argument_info = inspect.getargvalues(frame)
        argument_names = list(argument_info.args)
        if argument_info.varargs is not None:
            argument_names.append(argument_info.varargs)
        if argument_info.keywords is not None:
            argument_names.append(argument_info.keywords)
        self.hparams = {
            name: argument_info.locals[name]
            for name in argument_names
            if name not in set(ignore + ["self"]) and not name.startswith("_")
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
        for (k, v), ls, color in zip(  # noqa: B905
            self.data.items(),
            self.ls,
            self.colors,
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
            self.xlabel = "x"

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)

        display.display(self.fig)
        display.clear_output(wait=True)
