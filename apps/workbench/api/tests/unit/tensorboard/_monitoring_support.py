from __future__ import annotations


class TagsFailureAccumulator:
    def Tags(self):
        raise RuntimeError("broken tags")


class ReadFailureAccumulator:
    def Tags(self):
        return {
            "scalars": ["main_model.layers.0.model/output/mean"],
            "histograms": ["main_model.layers.0.model/histogram/usage_fraction"],
            "images": ["main_model.layers.0.model/heatmap/usage_fraction"],
        }

    def Scalars(self, tag):
        raise RuntimeError(f"broken scalar read: {tag}")

    def Histograms(self, tag):
        raise RuntimeError(f"broken histogram read: {tag}")

    def Images(self, tag):
        raise RuntimeError(f"broken image read: {tag}")


class FakeScalarEvent:
    def __init__(self, step: int, value: float) -> None:
        self.step = step
        self.value = value
        self.wall_time = float(step)


class NoMatchingMonitorAccumulator:
    def Tags(self):
        return {
            "scalars": ["other_node/output/mean"],
            "histograms": [],
            "images": [],
        }


class ParameterStatusAccumulator:
    def Tags(self):
        return {
            "scalars": ["main_model.layers.0.model/weights/relative_delta_norm"],
        }

    def Scalars(self, tag):
        if tag != "main_model.layers.0.model/weights/relative_delta_norm":
            raise KeyError(tag)
        return [FakeScalarEvent(1, 0.0), FakeScalarEvent(2, 0.5)]


class LargeParameterStatusAccumulator:
    def Tags(self):
        return {
            "scalars": [
                "main_model.layers.0.model/weights/relative_delta_norm",
                "main_model.layers.0.model/bias/l2_norm",
            ],
        }

    def Scalars(self, tag):
        if tag == "main_model.layers.0.model/weights/relative_delta_norm":
            return [
                FakeScalarEvent(1, 0.75),
                FakeScalarEvent(2, 0.0),
                FakeScalarEvent(3, 0.0),
                FakeScalarEvent(4, 0.0),
                FakeScalarEvent(5, 0.0),
            ]
        if tag == "main_model.layers.0.model/bias/l2_norm":
            return [
                FakeScalarEvent(1, 1.0),
                FakeScalarEvent(2, 2.0),
                FakeScalarEvent(3, 2.0),
                FakeScalarEvent(4, 2.0),
                FakeScalarEvent(5, 2.0),
            ]
        raise KeyError(tag)
