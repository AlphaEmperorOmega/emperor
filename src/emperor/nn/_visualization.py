import collections
import inspect


class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=None):
        """Save function arguments into class attribuites."""
        ignored_parameter_names = ignore or []
        instance_method_frame = inspect.currentframe()
        while instance_method_frame is not None:
            instance_method_frame = instance_method_frame.f_back
            if (
                instance_method_frame is not None
                and instance_method_frame.f_locals.get("self") is self
            ):
                break
        if instance_method_frame is None:
            raise RuntimeError(
                "save_hyperparameters must be called from an instance method."
            )
        caller_arguments = inspect.getargvalues(instance_method_frame)
        captured_argument_names = list(caller_arguments.args)
        if caller_arguments.varargs is not None:
            captured_argument_names.append(caller_arguments.varargs)
        if caller_arguments.keywords is not None:
            captured_argument_names.append(caller_arguments.keywords)
        self.hparams = {
            argument_name: caller_arguments.locals[argument_name]
            for argument_name in captured_argument_names
            if argument_name not in set(ignored_parameter_names + ["self"])
            and not argument_name.startswith("_")
        }
        for parameter_name, parameter_value in self.hparams.items():
            setattr(self, parameter_name, parameter_value)


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

        pending_points = self.raw_points[label]
        averaged_points = self.data[label]
        raw_point = Point(x, y)
        pending_points.append(raw_point)

        if len(pending_points) != every_n:
            return

        def coordinate_mean(coordinates):
            return sum(coordinates) / len(coordinates)

        mean_x_coordinate = coordinate_mean([point.x for point in pending_points])
        mean_y_coordinate = coordinate_mean([point.y for point in pending_points])
        averaged_point = Point(mean_x_coordinate, mean_y_coordinate)
        averaged_points.append(averaged_point)
        pending_points.clear()

        if not self.display:
            return

        import IPython.display as display
        import matplotlib.pyplot as plt

        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)

        plotted_lines, line_labels = [], []
        for (series_label, series_points), line_style, line_color in zip(  # noqa: B905
            self.data.items(),
            self.ls,
            self.colors,
        ):
            x_coordinates = [point.x for point in series_points]
            y_coordinates = [point.y for point in series_points]
            plotted_line = plt.plot(
                x_coordinates,
                y_coordinates,
                linestyle=line_style,
                color=line_color,
            )[0]
            plotted_lines.append(plotted_line)
            line_labels.append(series_label)

        plot_axes = self.axes if self.axes else plt.gca()
        if self.xlim:
            plot_axes.set_xlim(self.xlim)
        if self.ylim:
            plot_axes.set_ylim(self.ylim)
        if not self.xlabel:
            self.xlabel = "x"

        plot_axes.set_xlabel(self.xlabel)
        plot_axes.set_ylabel(self.ylabel)
        plot_axes.set_xscale(self.xscale)
        plot_axes.set_yscale(self.yscale)
        plot_axes.legend(plotted_lines, line_labels)

        display.display(self.fig)
        display.clear_output(wait=True)
