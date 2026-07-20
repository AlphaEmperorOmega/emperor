import unittest

import torch
from torch import nn

from emperor.neuron._optimizer_checkpoint import (
    LegacyOptimizerAppendPolicy,
    NeuronOptimizerCheckpointReconciler,
    NeuronOptimizerLoadTransaction,
)


class _DynamicCluster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cluster = nn.ModuleDict(
            {
                "neuron_0_0_0": nn.Linear(1, 1),
                "neuron_1_0_0": nn.Linear(1, 1),
            }
        )
        self.initial_x_axis_start = 0
        self.initial_x_axis_total_neurons = 1
        self.initial_y_axis_start = 0
        self.initial_y_axis_total_neurons = 1
        self.initial_z_axis_start = 0
        self.initial_z_axis_total_neurons = 1

    @staticmethod
    def _neuron_name(x_coordinate: int, y_coordinate: int, z_coordinate: int) -> str:
        return f"neuron_{x_coordinate}_{y_coordinate}_{z_coordinate}"


def _legacy_fixture():
    cluster = _DynamicCluster()
    optimizer = torch.optim.SGD(cluster.parameters(), lr=0.1, momentum=0.9)
    for sentinel, parameter in enumerate(cluster.parameters(), start=1):
        optimizer.state[parameter] = {
            "momentum_buffer": torch.full_like(parameter, float(sentinel))
        }
    serialized_state = optimizer.state_dict()
    serialized_ids_by_parameter_id = {
        id(parameter): serialized_id
        for parameter, serialized_id in zip(
            optimizer.param_groups[0]["params"],
            serialized_state["param_groups"][0]["params"],
            strict=True,
        )
    }
    base_parameters = list(cluster.cluster["neuron_0_0_0"].parameters())
    dynamic_parameters = list(cluster.cluster["neuron_1_0_0"].parameters())
    saved_options = {
        name: value
        for name, value in serialized_state["param_groups"][0].items()
        if name != "params"
    }

    def saved_group(parameters):
        return {
            **saved_options,
            "params": [
                serialized_ids_by_parameter_id[id(parameter)]
                for parameter in parameters
            ],
        }

    return (
        cluster,
        optimizer,
        {
            "state": serialized_state["state"],
            "param_groups": [
                saved_group(base_parameters),
                saved_group(dynamic_parameters),
            ],
        },
    )


class TestNeuronOptimizerCheckpointReconciliation(unittest.TestCase):
    def test_load_transaction_requires_every_expected_optimizer(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(3)]
        optimizers = [torch.optim.SGD([parameter], lr=0.1) for parameter in parameters]
        transaction = NeuronOptimizerLoadTransaction()

        transaction.prepare_for_load(optimizers[:2])
        transaction.mark_optimizer_loaded(optimizers[2])
        transaction.mark_optimizer_loaded(optimizers[0])

        with self.assertRaisesRegex(RuntimeError, "partial Neuron optimizer"):
            transaction.commit_loaded()

        transaction.mark_optimizer_loaded(optimizers[1])
        transaction.commit_loaded()
        self.assertFalse(transaction.optimizer_requires_completion(optimizers[0]))

    def test_legacy_groups_are_mapped_by_dynamic_parameter_identity(self) -> None:
        cluster, optimizer, saved_state = _legacy_fixture()
        parameters = list(cluster.parameters())
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [saved_state])
        self.assertEqual(len(optimizer.param_groups), 2)
        optimizer.load_state_dict(saved_state)
        policy = reconciler.complete_optimizer_load(optimizer)

        self.assertEqual(policy, LegacyOptimizerAppendPolicy(1, 0, (0, 0)))
        for sentinel, parameter in enumerate(parameters, start=1):
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                torch.full_like(parameter, float(sentinel)),
            )

    def test_cancelled_reconciliation_restores_original_group_objects(self) -> None:
        cluster, optimizer, saved_state = _legacy_fixture()
        original_groups = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameters = list(original_group["params"])
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [saved_state])
        self.assertEqual(len(optimizer.param_groups), 2)
        reconciler.clear()

        self.assertIs(optimizer.param_groups, original_groups)
        self.assertEqual(optimizer.param_groups, [original_group])
        self.assertEqual(
            [id(parameter) for parameter in original_group["params"]],
            [id(parameter) for parameter in original_parameters],
        )
