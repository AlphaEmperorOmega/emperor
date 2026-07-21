from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from IPython.core.displaypub import DisplayPublisher
from IPython.core.interactiveshell import InteractiveShell

from emperor.nn._visualization import HyperParameters, ProgressBoard


class _CapturedHyperParameters(HyperParameters):
    def __init__(
        self,
        visible: int,
        ignored: str,
        _private: str,
        *values: int,
        **named: int,
    ) -> None:
        transient_local = "not a constructor argument"
        self.save_hyperparameters(ignore=["ignored"])
        self.transient_local_after_save = transient_local


class _RecordingDisplayPublisher(DisplayPublisher):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict[str, object]] = []
        self.clear_waits: list[object] = []

    def publish(
        self,
        data,
        metadata=None,
        source=None,
        **kwargs,
    ) -> None:
        self.payloads.append(data)

    def clear_output(self, wait=False) -> None:
        self.clear_waits.append(wait)


def _draw_with_recording_publisher(
    board: ProgressBoard,
    x: float,
    y: float,
    label: str,
) -> _RecordingDisplayPublisher:
    shell = InteractiveShell.instance()
    original_publisher = shell.display_pub
    recording_publisher = _RecordingDisplayPublisher()
    shell.display_pub = recording_publisher
    try:
        board.draw(x, y, label)
    finally:
        shell.display_pub = original_publisher
    return recording_publisher


class NeuralNetworkVisualizationTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_save_hyperparameters_captures_only_non_private_function_arguments(
        self,
    ) -> None:
        captured = _CapturedHyperParameters(
            3,
            "ignored",
            "private",
            5,
            8,
            depth=13,
        )

        self.assertEqual(
            captured.hparams,
            {
                "visible": 3,
                "values": (5, 8),
                "named": {"depth": 13},
            },
        )
        self.assertEqual(captured.visible, 3)
        self.assertEqual(captured.values, (5, 8))
        self.assertEqual(captured.named, {"depth": 13})
        self.assertFalse(hasattr(captured, "ignored"))
        self.assertFalse(hasattr(captured, "_private"))
        self.assertNotIn("transient_local", captured.hparams)

    def test_save_hyperparameters_rejects_calls_outside_instance_methods(
        self,
    ) -> None:
        captured = HyperParameters()

        def capture_without_instance_method() -> None:
            captured.save_hyperparameters()

        with self.assertRaises(RuntimeError) as error:
            capture_without_instance_method()
        self.assertEqual(
            str(error.exception),
            "save_hyperparameters must be called from an instance method.",
        )

    def test_progress_board_defaults_are_materialized_without_shared_lists(
        self,
    ) -> None:
        first = ProgressBoard(display=False)
        second = ProgressBoard(display=False)

        self.assertEqual(first.ls, ["-", "--", "-.", ":"])
        self.assertEqual(first.colors, ["C0", "C1", "C2", "C3"])
        self.assertIsNot(first.ls, second.ls)
        self.assertIsNot(first.colors, second.colors)
        self.assertEqual(first.figsize, (3.5, 2.5))
        self.assertEqual(first.xscale, "linear")
        self.assertEqual(first.yscale, "linear")

    def test_draw_aggregates_exact_means_per_label_without_display(self) -> None:
        board = ProgressBoard(
            display=False,
            ls=["-"],
            colors=["C4"],
        )

        board.draw(1.0, 2.0, "loss", every_n=2)
        self.assertEqual(len(board.raw_points["loss"]), 1)
        self.assertEqual(board.data["loss"], [])

        board.draw(3.0, 6.0, "loss", every_n=2)
        board.draw(5.0, 10.0, "accuracy", every_n=1)

        self.assertEqual(board.raw_points["loss"], [])
        self.assertEqual(
            [(point.x, point.y) for point in board.data["loss"]],
            [(2.0, 4.0)],
        )
        self.assertEqual(
            [(point.x, point.y) for point in board.data["accuracy"]],
            [(5.0, 10.0)],
        )
        point = board.data["loss"][0]
        self.assertEqual(type(point).__name__, "Point")
        self.assertEqual(point._fields, ("x", "y"))
        self.assertEqual(list(board.data), ["loss", "accuracy"])

    def test_default_visible_board_draws_real_figure_with_fallback_x_label(
        self,
    ) -> None:
        board = ProgressBoard(display=True)

        publisher = _draw_with_recording_publisher(board, 1.0, 2.0, "metric")

        self.assertIsNotNone(board.fig)
        self.assertEqual(tuple(board.fig.get_size_inches()), (3.5, 2.5))
        self.assertEqual(board.xlabel, "x")
        axes = board.fig.axes[0]
        self.assertEqual(axes.get_xlabel(), "x")
        self.assertEqual(axes.get_xscale(), "linear")
        self.assertEqual(axes.get_yscale(), "linear")
        self.assertEqual(
            [text.get_text() for text in axes.get_legend().get_texts()],
            ["metric"],
        )
        self.assertEqual(publisher.clear_waits, [True])
        self.assertEqual(len(publisher.payloads), 1)
        self.assertIn(
            "Figure size 350x250",
            publisher.payloads[0]["text/plain"],
        )

    def test_visible_board_honors_supplied_axes_limits_scales_and_styles(
        self,
    ) -> None:
        figure, axes = plt.subplots()
        axes.plot([1.0], [0.0], color="black", label="preexisting")
        board = ProgressBoard(
            xlabel="step",
            ylabel="score",
            xlim=(1.0, 10.0),
            ylim=(-2.0, 2.0),
            xscale="log",
            yscale="linear",
            ls=[":"],
            colors=["C3"],
            fig=figure,
            axes=axes,
            display=True,
        )

        publisher = _draw_with_recording_publisher(board, 2.0, 1.5, "score")

        self.assertIs(board.fig, figure)
        self.assertIs(board.axes, axes)
        self.assertEqual(axes.get_xlabel(), "step")
        self.assertEqual(axes.get_ylabel(), "score")
        self.assertEqual(axes.get_xlim(), (1.0, 10.0))
        self.assertEqual(axes.get_ylim(), (-2.0, 2.0))
        self.assertEqual(axes.get_xscale(), "log")
        self.assertEqual(axes.lines[1].get_linestyle(), ":")
        self.assertEqual(axes.lines[1].get_color(), "C3")
        self.assertEqual(axes.lines[1].get_xdata().tolist(), [2.0])
        self.assertEqual(axes.lines[1].get_ydata().tolist(), [1.5])
        legend = axes.get_legend()
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()],
            ["score"],
        )
        self.assertEqual(legend.legend_handles[0].get_color(), "C3")
        self.assertEqual(publisher.clear_waits, [True])


if __name__ == "__main__":
    unittest.main()
