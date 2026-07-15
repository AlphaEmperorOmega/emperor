import torch
import torch.nn as nn
from emperor.neuron import NeuronClusterConfig
from emperor.neuron.core.monitor import NeuronClusterMonitorCallback

from support.monitor import orchestration_calls
from unit.test_neuron import NeuronTestCase


class FakeExperiment:
    def __init__(self):
        self.histograms = []
        self.images = []

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values.clone(), step))

    def add_image(self, tag, image, step, dataformats):
        self.images.append((tag, image.clone(), step, dataformats))


class FakeLogger:
    def __init__(self, experiment):
        self.experiment = experiment


class FakeLightningModule(nn.Module):
    def __init__(self, cluster, experiment=None, global_step: int = 0):
        super().__init__()
        self.neuron_cluster = cluster
        self.logger = FakeLogger(experiment) if experiment is not None else None
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


class TestNeuronClusterMonitorCallback(NeuronTestCase):
    CLUSTER_PATH = "neuron_cluster"

    def test_tracking_orchestration_lists_each_tracked_fact(self):
        cls = NeuronClusterMonitorCallback
        orchestration = (
            cls._NeuronClusterMonitorCallback__track_neuron_cluster_diagnostics
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_neuron_count",
                "__track_cluster_capacity",
                "__track_cluster_fill_fraction",
                "__track_growth_events",
                "__track_pruning_events",
                "__track_growth_pressure_mean",
                "__track_growth_pressure_maximum",
                "__track_total_growths",
                "__track_growth_budget_remaining",
                "__track_growth_cooldown_remaining",
                "__track_pruning_pressure_mean",
                "__track_pruning_pressure_maximum",
                "__track_auxiliary_loss",
                "__track_route_depth_mean",
                "__track_route_depth_maximum",
                "__track_recurrent_steps",
                "__track_escape_fraction",
                "__track_valid_fraction",
                "__track_halted_fraction",
                "__track_active_neuron_count",
                "__track_entry_routing_entropy",
                "__track_marginal_entry_routing_entropy",
                "__track_routing_coefficient_of_variation",
                "__track_survival_history",
                "__track_route_depth_histogram",
                "__track_survival_heatmap",
                "__track_neuron_utilization_heatmap",
            ),
        )

    def build_cluster(
        self,
        x_axis_total_neurons: int = 2,
        y_axis_total_neurons: int = 2,
        z_axis_total_neurons: int = 1,
        max_steps: int = 2,
        beam_width: int | None = None,
        growth_threshold: int | None = None,
        growth_cooldown_steps: int | None = None,
        max_total_growths: int | None = None,
        pruning_threshold: int | None = None,
    ):
        return NeuronClusterConfig(
            x_axis_total_neurons=x_axis_total_neurons,
            y_axis_total_neurons=y_axis_total_neurons,
            z_axis_total_neurons=z_axis_total_neurons,
            max_steps=max_steps,
            beam_width=beam_width,
            growth_threshold=growth_threshold,
            growth_cooldown_steps=growth_cooldown_steps,
            max_total_growths=max_total_growths,
            pruning_threshold=pruning_threshold,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def build_module(self, cluster, experiment=None, global_step: int = 0):
        return FakeLightningModule(
            cluster, experiment=experiment, global_step=global_step
        )

    def primed_callback(self, module, **callback_kwargs):
        callback = NeuronClusterMonitorCallback(**callback_kwargs)
        callback.on_fit_start(trainer=None, pl_module=module)
        return callback

    def feed_cluster(self, module) -> None:
        module.neuron_cluster(torch.randn(self.batch_size, self.input_dim))

    def scalar_names(self, module) -> set:
        return {name for name, _ in module.logged_scalars}

    def route_scalar_suffixes(self) -> tuple:
        return (
            "route/depth_mean",
            "route/depth_max",
            "route/recurrent_steps",
            "route/escape_fraction",
            "route/valid_fraction",
            "route/halted_fraction",
            "route/active_neuron_count",
            "entry/routing_entropy",
            "entry/routing_entropy_marginal",
            "entry/routing_coefficient_of_variation",
            "loss/auxiliary_loss",
        )

    def test_rejects_non_positive_log_interval(self):
        with self.assertRaises(ValueError):
            NeuronClusterMonitorCallback(log_every_n_steps=0)

    def test_rejects_non_positive_history_size(self):
        with self.assertRaises(ValueError):
            NeuronClusterMonitorCallback(history_size=0)

    def test_on_fit_start_discovers_and_wraps_clusters(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster)

        callback = self.primed_callback(module)

        self.assertEqual(len(callback._clusters), 1)
        attached_name, attached_cluster = callback._clusters[0]
        self.assertEqual(attached_name, self.CLUSTER_PATH)
        self.assertIs(attached_cluster, cluster)
        self.assertIn("forward", cluster.__dict__)
        self.assertIn(self.CLUSTER_PATH, callback._survival_history)

    def test_repeated_fit_start_replaces_existing_forward_wrapper(self):
        cluster = self.build_cluster()
        original_forward = cluster.forward
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)

        callback.on_fit_start(trainer=None, pl_module=module)
        self.feed_cluster(module)
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        self.assertEqual(len(callback._forward_replacements), 1)
        depth_tag = f"{self.CLUSTER_PATH}/cluster/route/depth_mean"
        self.assertEqual(
            [name for name, _ in module.logged_scalars].count(depth_tag),
            1,
        )
        callback.on_fit_end(trainer=None, pl_module=module)
        self.assertEqual(cluster.forward, original_forward)

    def test_cleanup_restores_preexisting_instance_forward(self):
        cluster = self.build_cluster()
        class_forward = cluster.forward

        def instance_forward(input_tensor, return_trace=False):
            return class_forward(input_tensor, return_trace=return_trace)

        cluster.forward = instance_forward
        module = self.build_module(cluster)
        callback = self.primed_callback(module)

        callback.on_fit_end(trainer=None, pl_module=module)

        self.assertIs(cluster.__dict__["forward"], instance_forward)

    def test_beam_cluster_is_not_wrapped_and_logs_only_structural_scalars(self):
        cluster = self.build_cluster(beam_width=2)
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)

        self.feed_cluster(module)
        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        self.assertNotIn("forward", cluster.__dict__)
        names = self.scalar_names(module)
        for suffix in ("neuron_count", "capacity", "fill_fraction"):
            self.assertIn(f"{self.CLUSTER_PATH}/cluster/{suffix}", names)
        for suffix in self.route_scalar_suffixes():
            self.assertNotIn(f"{self.CLUSTER_PATH}/cluster/{suffix}", names)

    def test_wrapped_forward_returns_normal_output_tuple(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        self.primed_callback(module, log_every_n_steps=1)

        output = cluster(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, (self.batch_size, self.input_dim))

    def test_wrapped_forward_honours_explicit_return_trace(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        self.primed_callback(module, log_every_n_steps=1)

        output, auxiliary_loss, trace = cluster(
            torch.randn(self.batch_size, self.input_dim), return_trace=True
        )

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertIsNotNone(trace)

    def test_eval_forward_does_not_capture_trace(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        module.eval()

        with torch.no_grad():
            self.feed_cluster(module)

        self.assertNotIn(self.CLUSTER_PATH, callback._latest_observations)

    def test_on_train_batch_end_skips_when_not_at_logging_interval(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster, global_step=3)
        callback = self.primed_callback(module, log_every_n_steps=10)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        self.assertEqual(module.logged_scalars, [])

    def test_logs_size_and_route_scalars(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        names = self.scalar_names(module)
        for suffix in (
            "neuron_count",
            "capacity",
            "fill_fraction",
            "growth/events",
            "pruning/events",
        ):
            self.assertIn(f"{self.CLUSTER_PATH}/cluster/{suffix}", names)
        for suffix in self.route_scalar_suffixes():
            self.assertIn(f"{self.CLUSTER_PATH}/cluster/{suffix}", names)

    def test_growth_pressure_logged_when_threshold_set(self):
        cluster = self.build_cluster(growth_threshold=2)
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        names = self.scalar_names(module)
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/growth/pressure_mean", names)
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/growth/pressure_max", names)

    def test_growth_pressure_skipped_when_threshold_disabled(self):
        cluster = self.build_cluster(growth_threshold=None)
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        for name, _ in module.logged_scalars:
            self.assertNotIn("growth/pressure", name)

    def test_growth_budget_logged_when_options_set(self):
        cluster = self.build_cluster(
            growth_threshold=2,
            growth_cooldown_steps=3,
            max_total_growths=4,
        )
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        names = self.scalar_names(module)
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/growth/total_growths", names)
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/growth/budget_remaining", names)
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/growth/cooldown_remaining", names)

    def test_growth_budget_skipped_when_options_disabled(self):
        cluster = self.build_cluster(growth_threshold=2)
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        for name, _ in module.logged_scalars:
            self.assertNotIn("growth/total_growths", name)
            self.assertNotIn("growth/budget_remaining", name)
            self.assertNotIn("growth/cooldown_remaining", name)

    def test_pruning_pressure_logged_when_threshold_set(self):
        cluster = self.build_cluster(pruning_threshold=2)
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        names = self.scalar_names(module)
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/pruning/pressure_mean", names)
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/pruning/pressure_max", names)

    def test_pruning_pressure_skipped_when_threshold_disabled(self):
        cluster = self.build_cluster(pruning_threshold=None)
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        for name, _ in module.logged_scalars:
            self.assertNotIn("pruning/pressure", name)

    def test_visual_summaries_logged_when_experiment_present(self):
        experiment = FakeExperiment()
        cluster = self.build_cluster()
        module = self.build_module(cluster, experiment=experiment, global_step=0)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        histogram_tags = {tag for tag, _, _ in experiment.histograms}
        self.assertIn(
            f"{self.CLUSTER_PATH}/cluster/histogram/route_depth", histogram_tags
        )
        image_tags = {tag for tag, _, _, _ in experiment.images}
        self.assertIn(f"{self.CLUSTER_PATH}/cluster/heatmap/survival", image_tags)
        self.assertIn(
            f"{self.CLUSTER_PATH}/cluster/heatmap/neuron_utilization", image_tags
        )

    def test_visual_summaries_skipped_when_no_experiment(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster, experiment=None)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        self.assertEqual(len(callback._survival_history[self.CLUSTER_PATH]), 0)

    def test_survival_history_bounded_by_history_size(self):
        experiment = FakeExperiment()
        cluster = self.build_cluster()
        module = self.build_module(cluster, experiment=experiment)
        callback = self.primed_callback(module, log_every_n_steps=1, history_size=3)

        for _ in range(5):
            self.feed_cluster(module)
            callback.on_train_batch_end(
                trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
            )

        history = callback._survival_history[self.CLUSTER_PATH]
        self.assertEqual(len(history), 3)
        for tensor in history.tensors:
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)

    def test_route_observation_is_consumed_once(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_cluster(module)

        for _ in range(2):
            callback.on_train_batch_end(
                trainer=None,
                pl_module=module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )

        depth_tag = f"{self.CLUSTER_PATH}/cluster/route/depth_mean"
        self.assertEqual(
            [name for name, _ in module.logged_scalars].count(depth_tag),
            1,
        )

    def test_on_fit_end_restores_forward_and_clears_state(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        callback = self.primed_callback(module)

        callback.on_fit_end(trainer=None, pl_module=module)

        self.assertNotIn("forward", cluster.__dict__)
        self.assertEqual(callback._clusters, [])
        self.assertEqual(callback._survival_history, {})
        self.assertEqual(callback._latest_observations, {})
        self.assertEqual(callback._previous_neuron_names, {})


if __name__ == "__main__":
    import unittest

    unittest.main()
