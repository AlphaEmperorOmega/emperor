import copy
import math

import torch
import torch.nn as nn

from emperor.neuron import (
    NeuronClusterConfig,
    NeuronClusterMonitorCallback,
    NeuronClusterTrace,
    NeuronClusterTraceStep,
)
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
        module.global_step += 1

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

    def controlled_trace(self) -> NeuronClusterTrace:
        entry_probabilities = torch.tensor(
            [
                [0.2, 0.1, 0.1],
                [0.1, 0.3, 0.0],
            ],
            dtype=torch.float32,
        ).repeat(3, 1)
        entry_selected_coordinates = torch.tensor(
            [
                [[1, 1, 1], [1, 1, 1], [4, 1, 1]],
                [[1, 2, 1], [2, 1, 1], [4, 1, 1]],
                [[1, 1, 1], [2, 2, 1], [4, 1, 1]],
                [[3, 2, 1], [3, 2, 1], [4, 1, 1]],
                [[4, 1, 1], [1, 1, 1], [1, 2, 1]],
                [[2, 1, 1], [4, 1, 1], [3, 2, 1]],
            ],
            dtype=torch.long,
        )
        entry_valid_mask = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [False, True, True],
                [True, False, True],
            ]
        )
        recurrent_selected_coordinates = torch.tensor(
            [
                [[1, 1, 1], [1, 1, 1], [4, 1, 1]],
                [[1, 1, 1], [2, 2, 1], [4, 1, 1]],
                [[1, 1, 1], [1, 2, 1], [4, 1, 1]],
                [[1, 1, 1], [4, 1, 1], [3, 2, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.long,
        )
        recurrent_valid_mask = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, False, True],
                [False, False, False],
                [False, False, False],
            ]
        )
        recurrent_escape_mask = torch.tensor(
            [
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ]
        )
        recurrent_step = NeuronClusterTraceStep(
            probabilities=torch.cat(
                (entry_probabilities[:4], torch.zeros(2, 3)),
                dim=0,
            ),
            selected_coordinates=recurrent_selected_coordinates,
            valid_mask=recurrent_valid_mask,
            escape_mask=recurrent_escape_mask,
            chosen_branch_indices=torch.tensor([0, 1, 0, 1, 0, 0]),
            halt_mask=torch.tensor([False, True, False, True, True, False]),
            active_mask=torch.tensor([True, True, True, False, False, False]),
        )
        return NeuronClusterTrace(
            input_shape=(6, self.input_dim),
            entry_coordinates=torch.tensor(
                [
                    [1, 1, 1],
                    [1, 2, 1],
                    [2, 1, 1],
                    [2, 2, 1],
                    [3, 1, 1],
                    [3, 2, 1],
                ]
            ),
            entry_probabilities=entry_probabilities,
            entry_selected_coordinates=entry_selected_coordinates,
            entry_valid_mask=entry_valid_mask,
            entry_escape_mask=~entry_valid_mask,
            entry_chosen_branch_indices=torch.tensor([0, 1, 0, 1, 0, 1]),
            entry_halt_mask=torch.zeros(6, dtype=torch.bool),
            entry_active_mask=torch.tensor([True, True, True, True, False, False]),
            steps=[recurrent_step],
        )

    def test_rejects_non_positive_log_interval(self):
        with self.assertRaises(ValueError):
            NeuronClusterMonitorCallback(log_every_n_steps=0)

    def test_rejects_non_positive_history_size(self):
        with self.assertRaises(ValueError):
            NeuronClusterMonitorCallback(history_size=0)

    def test_rejects_non_integer_cadence_and_history_options(self) -> None:
        for option_name in ("log_every_n_steps", "history_size"):
            for value in (True, False, 1.0, 1.5, "1"):
                with self.subTest(option_name=option_name, value=value):
                    with self.assertRaisesRegex(
                        TypeError,
                        rf"^{option_name} must be a positive integer",
                    ):
                        NeuronClusterMonitorCallback(**{option_name: value})

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

    def test_second_monitor_cannot_install_a_nested_forward_wrapper(self):
        cluster = self.build_cluster()
        original_forward = cluster.forward
        module = self.build_module(cluster)
        first = self.primed_callback(module, log_every_n_steps=1)
        first_wrapper = cluster.forward
        second = NeuronClusterMonitorCallback(log_every_n_steps=1)

        with self.assertRaisesRegex(RuntimeError, "already monitored"):
            second.on_fit_start(trainer=None, pl_module=module)

        self.assertIs(cluster.forward, first_wrapper)
        second.on_fit_end(trainer=None, pl_module=module)
        self.assertIs(cluster.forward, first_wrapper)
        first.on_fit_end(trainer=None, pl_module=module)
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

    def test_wrapped_forward_preserves_public_input_keyword(self) -> None:
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        self.primed_callback(module, log_every_n_steps=1)
        input_tensor = torch.randn(self.batch_size, self.input_dim)

        output, auxiliary_loss = cluster(input=input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(auxiliary_loss.shape, ())

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

    def test_interval_uses_the_same_training_step_for_capture_and_emission(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster, global_step=0)
        callback = self.primed_callback(module, log_every_n_steps=2)

        self.feed_cluster(module)
        module.global_step = 1
        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )
        self.feed_cluster(module)
        module.global_step = 2
        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=1
        )

        depth_tag = f"{self.CLUSTER_PATH}/cluster/route/depth_mean"
        self.assertEqual(
            [name for name, _ in module.logged_scalars].count(depth_tag),
            1,
        )

    def test_gradient_accumulation_does_not_emit_before_optimizer_step(self):
        cluster = self.build_cluster()
        module = self.build_module(cluster, global_step=0)
        callback = self.primed_callback(module, log_every_n_steps=1)

        cluster(torch.randn(self.batch_size, self.input_dim))
        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )
        cluster(torch.randn(self.batch_size, self.input_dim))
        module.global_step = 1
        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=1
        )

        depth_tag = f"{self.CLUSTER_PATH}/cluster/route/depth_mean"
        self.assertEqual(
            [name for name, _ in module.logged_scalars].count(depth_tag),
            1,
        )

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
        growth_tag = f"{self.CLUSTER_PATH}/cluster/growth/events"
        self.assertEqual(
            [name for name, _ in module.logged_scalars].count(growth_tag),
            1,
        )

    def test_logs_exact_capacity_and_fill_fraction_for_sparse_cluster(self) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=4,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        module.global_step = 1

        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        scalar_values = dict(module.logged_scalars)
        self.assertEqual(
            scalar_values[f"{self.CLUSTER_PATH}/cluster/neuron_count"],
            2.0,
        )
        self.assertEqual(
            scalar_values[f"{self.CLUSTER_PATH}/cluster/capacity"],
            4.0,
        )
        self.assertEqual(
            scalar_values[f"{self.CLUSTER_PATH}/cluster/fill_fraction"],
            0.5,
        )

    def test_logs_exact_structural_and_plasticity_metrics_once_per_step(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=3,
            z_axis_total_neurons=4,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=10,
            growth_cooldown_steps=5,
            max_total_growths=7,
            pruning_threshold=8,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        for neuron, batch_count, atrophy_count in zip(
            cluster.cluster.values(),
            (2, 8),
            (1, 5),
            strict=True,
        ):
            neuron.batch_counter.fill_(batch_count)
            neuron.atrophy_counter.fill_(atrophy_count)
        cluster.total_growth_count.fill_(2)
        cluster.forwards_since_last_growth.fill_(7)
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=1)
        module.global_step = 1

        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        expected_scalars = {
            "neuron_count": 2.0,
            "capacity": 24.0,
            "fill_fraction": 1.0 / 12.0,
            "growth/pressure_mean": 0.5,
            "growth/pressure_max": 0.8,
            "pruning/pressure_mean": 0.375,
            "pruning/pressure_max": 0.625,
            "growth/total_growths": 2.0,
            "growth/budget_remaining": 5.0,
            "growth/cooldown_remaining": 0.0,
        }
        scalar_values = dict(module.logged_scalars)
        for suffix, expected_value in expected_scalars.items():
            with self.subTest(suffix=suffix):
                torch.testing.assert_close(
                    torch.as_tensor(
                        scalar_values[f"{self.CLUSTER_PATH}/cluster/{suffix}"]
                    ).float(),
                    torch.tensor(expected_value, dtype=torch.float32),
                    rtol=0.0,
                    atol=0.0,
                )

        emitted_scalar_count = len(module.logged_scalars)
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )
        self.assertEqual(len(module.logged_scalars), emitted_scalar_count)

        singleton_cluster = self.build_cluster(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
        )
        singleton_module = self.build_module(singleton_cluster)
        singleton_callback = self.primed_callback(
            singleton_module,
            log_every_n_steps=1,
        )
        singleton_module.global_step = 1
        singleton_callback.on_train_batch_end(
            trainer=None,
            pl_module=singleton_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )
        singleton_scalars = dict(singleton_module.logged_scalars)
        self.assertEqual(
            singleton_scalars[f"{self.CLUSTER_PATH}/cluster/neuron_count"],
            1.0,
        )
        self.assertEqual(
            singleton_scalars[f"{self.CLUSTER_PATH}/cluster/capacity"],
            1.0,
        )
        self.assertEqual(
            singleton_scalars[f"{self.CLUSTER_PATH}/cluster/fill_fraction"],
            1.0,
        )

    def test_interval_counts_prune_and_regrow_of_the_same_name(self) -> None:
        cluster = self.build_cluster()
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=2)
        neuron_name = next(iter(cluster.cluster))
        removed_neuron = cluster.cluster[neuron_name]

        del cluster.cluster[neuron_name]
        module.global_step = 1
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )
        cluster.cluster[neuron_name] = removed_neuron
        module.global_step = 2
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )

        scalar_values = dict(module.logged_scalars)
        self.assertEqual(
            scalar_values[f"{self.CLUSTER_PATH}/cluster/growth/events"],
            1.0,
        )
        self.assertEqual(
            scalar_values[f"{self.CLUSTER_PATH}/cluster/pruning/events"],
            1.0,
        )

    def test_topology_events_accumulate_and_reset_across_logging_intervals(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=4,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        module = self.build_module(cluster)
        callback = self.primed_callback(module, log_every_n_steps=3)

        traced_output = cluster(
            torch.randn(self.batch_size, self.input_dim),
            return_trace=True,
        )
        self.assertEqual(len(traced_output), 3)
        self.assertIsNotNone(traced_output[2])

        def finish_batch(global_step: int) -> None:
            module.global_step = global_step
            callback.on_train_batch_end(
                trainer=None,
                pl_module=module,
                outputs=None,
                batch=None,
                batch_idx=global_step - 1,
            )

        template_neuron = next(iter(cluster.cluster.values()))
        first_added_name = "neuron_1_1_1"
        second_added_name = "neuron_3_1_1"
        cluster.cluster[first_added_name] = copy.deepcopy(template_neuron)
        finish_batch(1)
        cluster.cluster[second_added_name] = copy.deepcopy(template_neuron)
        finish_batch(2)
        finish_batch(3)

        del cluster.cluster[first_added_name]
        finish_batch(4)
        finish_batch(5)
        finish_batch(6)

        finish_batch(7)
        finish_batch(8)
        finish_batch(9)

        growth_tag = f"{self.CLUSTER_PATH}/cluster/growth/events"
        pruning_tag = f"{self.CLUSTER_PATH}/cluster/pruning/events"
        growth_events = [
            value for name, value in module.logged_scalars if name == growth_tag
        ]
        pruning_events = [
            value for name, value in module.logged_scalars if name == pruning_tag
        ]
        self.assertEqual(growth_events, [2.0, 0.0, 0.0])
        self.assertEqual(pruning_events, [0.0, 1.0, 0.0])

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

    def test_controlled_trace_emits_exact_route_entry_and_visual_diagnostics(
        self,
    ) -> None:
        experiment = FakeExperiment()
        cluster = self.build_cluster(
            x_axis_total_neurons=3,
            y_axis_total_neurons=2,
            z_axis_total_neurons=1,
        )
        trace = self.controlled_trace()

        def deterministic_forward(input_tensor, return_trace=False):
            output = input_tensor + 1.0
            auxiliary_loss = input_tensor.new_tensor(0.625)
            if return_trace:
                return output, auxiliary_loss, trace
            return output, auxiliary_loss

        cluster.forward = deterministic_forward
        module = self.build_module(cluster, experiment=experiment)
        callback = self.primed_callback(module, log_every_n_steps=1)

        output, auxiliary_loss = cluster(torch.zeros(6, self.input_dim))
        module.global_step = 1
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        torch.testing.assert_close(output, torch.ones_like(output))
        self.assertEqual(float(auxiliary_loss), 0.625)
        entropy_for_first_distribution = 1.5 * math.log(2.0)
        entropy_for_second_distribution = -(
            0.25 * math.log(0.25) + 0.75 * math.log(0.75)
        )
        expected_mean_entry_entropy = (
            entropy_for_first_distribution + entropy_for_second_distribution
        ) / 2.0
        expected_marginal_entry_entropy = -(
            0.375 * math.log(0.375) + 0.5 * math.log(0.5) + 0.125 * math.log(0.125)
        )
        expected_scalars = {
            "route/depth_mean": 7.0 / 6.0,
            "route/depth_max": 2.0,
            "route/recurrent_steps": 1.0,
            "route/escape_fraction": 5.0 / 18.0,
            "route/valid_fraction": 5.0 / 9.0,
            "route/halted_fraction": 0.5,
            "route/active_neuron_count": 5.0,
            "entry/routing_entropy": expected_mean_entry_entropy,
            "entry/routing_entropy_marginal": expected_marginal_entry_entropy,
            "entry/routing_coefficient_of_variation": math.sqrt(7.0 / 32.0),
            "loss/auxiliary_loss": 0.625,
        }
        scalar_values = dict(module.logged_scalars)
        for suffix, expected_value in expected_scalars.items():
            with self.subTest(suffix=suffix):
                torch.testing.assert_close(
                    torch.as_tensor(
                        scalar_values[f"{self.CLUSTER_PATH}/cluster/{suffix}"]
                    ).float(),
                    torch.tensor(expected_value, dtype=torch.float32),
                    rtol=1e-6,
                    atol=1e-7,
                )

        route_depth_tag = f"{self.CLUSTER_PATH}/cluster/histogram/route_depth"
        histogram_by_tag = {
            tag: (values, step) for tag, values, step in experiment.histograms
        }
        route_depth, histogram_step = histogram_by_tag[route_depth_tag]
        torch.testing.assert_close(
            route_depth,
            torch.tensor([2.0, 2.0, 2.0, 1.0, 0.0, 0.0]),
        )
        self.assertEqual(histogram_step, 1)

        image_by_tag = {
            tag: (image, step, dataformats)
            for tag, image, step, dataformats in experiment.images
        }
        survival_heatmap, survival_step, survival_dataformats = image_by_tag[
            f"{self.CLUSTER_PATH}/cluster/heatmap/survival"
        ]
        torch.testing.assert_close(
            survival_heatmap,
            torch.tensor([[[2.0 / 3.0], [0.5]]]),
        )
        self.assertEqual((survival_step, survival_dataformats), (1, "CHW"))
        utilization_heatmap, utilization_step, utilization_dataformats = image_by_tag[
            f"{self.CLUSTER_PATH}/cluster/heatmap/neuron_utilization"
        ]
        torch.testing.assert_close(
            utilization_heatmap,
            torch.tensor(
                [
                    [
                        [1.0, 1.0 / 3.0],
                        [2.0 / 9.0, 2.0 / 9.0],
                        [0.0, 4.0 / 9.0],
                    ]
                ]
            ),
        )
        self.assertEqual((utilization_step, utilization_dataformats), (1, "CHW"))

        emitted_media_counts = (len(experiment.histograms), len(experiment.images))
        route_scalar_count = sum(
            name.endswith("/cluster/route/depth_mean")
            for name, _ in module.logged_scalars
        )
        module.global_step = 2
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )

        self.assertEqual(
            (len(experiment.histograms), len(experiment.images)),
            emitted_media_counts,
        )
        self.assertEqual(
            sum(
                name.endswith("/cluster/route/depth_mean")
                for name, _ in module.logged_scalars
            ),
            route_scalar_count,
        )

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
        self.assertEqual(callback._pending_growth_events, {})
        self.assertEqual(callback._pending_pruning_events, {})

    def test_exception_cleanup_restores_forward_and_releases_monitor_owner(self):
        cluster = self.build_cluster()
        original_forward = cluster.forward
        module = self.build_module(cluster)
        callback = self.primed_callback(module)

        callback.on_exception(
            trainer=None,
            pl_module=module,
            exception=RuntimeError("training failed"),
        )

        self.assertEqual(cluster.forward, original_forward)
        self.assertNotIn(callback._OWNER_ATTRIBUTE, cluster.__dict__)
        replacement = NeuronClusterMonitorCallback()
        replacement.on_fit_start(trainer=None, pl_module=module)
        replacement.on_fit_end(trainer=None, pl_module=module)

    def test_cleanup_restores_forward_without_deleting_foreign_ownership(self):
        cluster = self.build_cluster()
        original_forward = cluster.forward
        module = self.build_module(cluster)
        callback = self.primed_callback(module)
        foreign_owner = object()
        cluster.__dict__[callback._OWNER_ATTRIBUTE] = foreign_owner

        callback.on_fit_end(trainer=None, pl_module=module)

        self.assertEqual(cluster.forward, original_forward)
        self.assertIs(
            cluster.__dict__[callback._OWNER_ATTRIBUTE],
            foreign_owner,
        )

    def test_utilization_grid_ignores_empty_and_out_of_range_coordinates(self):
        accumulate = NeuronClusterMonitorCallback._NeuronClusterMonitorCallback__accumulate_coordinate_counts
        cases = (
            (
                torch.tensor([[[1, 1, 1]]]),
                torch.tensor([[False]]),
            ),
            (
                torch.tensor([[[0, 1, 1], [3, 1, 1]]]),
                torch.tensor([[True, True]]),
            ),
        )
        for coordinates, valid_mask in cases:
            with self.subTest(coordinates=coordinates.tolist()):
                utilization_grid = torch.zeros(2, 2)

                accumulate(utilization_grid, coordinates, valid_mask)

                torch.testing.assert_close(
                    utilization_grid,
                    torch.zeros_like(utilization_grid),
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
