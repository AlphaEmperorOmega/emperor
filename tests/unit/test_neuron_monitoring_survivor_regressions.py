import unittest

import torch
import torch.nn as nn

from emperor.neuron import (
    NeuronClusterConfig,
    NeuronClusterMonitorCallback,
    NeuronClusterTrace,
)
from emperor.neuron._monitoring.diagnostics import _NeuronDiagnostics
from unit.test_neuron import NeuronTestCase


class _RecordingLightningModule(nn.Module):
    def __init__(self, cluster: nn.Module, global_step: int) -> None:
        super().__init__()
        self.neuron_cluster = cluster
        self.global_step = global_step
        self.logger = None
        self.logged_scalars: list[tuple[str, object]] = []

    def log(self, name: str, value: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.logged_scalars.append((name, value))


class TestNeuronMonitoringSurvivorRegressions(NeuronTestCase):
    def test_resumed_fit_uses_global_step_and_captures_the_next_third_step(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=2,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        module = _RecordingLightningModule(cluster, global_step=6)
        callback = NeuronClusterMonitorCallback(log_every_n_steps=3)
        callback.on_fit_start(trainer=None, pl_module=module)

        try:
            callback.on_train_batch_end(
                trainer=None,
                pl_module=module,
                outputs=None,
                batch=None,
                batch_idx=5,
            )
            self.assertEqual(module.logged_scalars, [])

            module.global_step = 8
            cluster(torch.randn(self.batch_size, self.input_dim))
            module.global_step = 9
            callback.on_train_batch_end(
                trainer=None,
                pl_module=module,
                outputs=None,
                batch=None,
                batch_idx=8,
            )

            route_depth_tag = "neuron_cluster/cluster/route/depth_mean"
            self.assertEqual(
                [name for name, _ in module.logged_scalars].count(route_depth_tag),
                1,
            )
        finally:
            callback.on_fit_end(trainer=None, pl_module=module)

    def test_utilization_grid_rejects_coordinates_at_both_upper_bounds(
        self,
    ) -> None:
        accumulate = NeuronClusterMonitorCallback._NeuronClusterMonitorCallback__accumulate_coordinate_counts
        utilization_grid = torch.zeros(2, 2)
        upper_bound_coordinates = torch.tensor(
            [[[1, 3, 1], [3, 1, 1]]],
            dtype=torch.long,
        )

        accumulate(
            utilization_grid,
            upper_bound_coordinates,
            torch.tensor([[True, True]]),
        )

        torch.testing.assert_close(
            utilization_grid,
            torch.zeros_like(utilization_grid),
        )

    def test_utilization_grid_counts_one_valid_coordinate(self) -> None:
        accumulate = NeuronClusterMonitorCallback._NeuronClusterMonitorCallback__accumulate_coordinate_counts
        utilization_grid = torch.zeros(2, 2)

        accumulate(
            utilization_grid,
            torch.tensor([[[2, 2, 1]]], dtype=torch.long),
            torch.tensor([[True]]),
        )

        torch.testing.assert_close(
            utilization_grid,
            torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        )

    def test_route_diagnostics_include_a_single_valid_mask_element(self) -> None:
        trace = NeuronClusterTrace(
            input_shape=(1, self.input_dim),
            entry_coordinates=torch.tensor([[1, 1, 1]], dtype=torch.long),
            entry_probabilities=torch.ones(1, 1),
            entry_selected_coordinates=torch.tensor(
                [[[1, 1, 1]]],
                dtype=torch.long,
            ),
            entry_valid_mask=torch.tensor([[True]]),
            entry_escape_mask=torch.tensor([[False]]),
            entry_chosen_branch_indices=torch.zeros(1, dtype=torch.long),
            entry_halt_mask=torch.tensor([False]),
            entry_active_mask=torch.tensor([True]),
        )

        metrics = _NeuronDiagnostics.calculate_route(trace)

        torch.testing.assert_close(metrics.valid_fraction, torch.tensor(1.0))
        torch.testing.assert_close(metrics.escape_fraction, torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
