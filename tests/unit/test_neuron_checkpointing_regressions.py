import re
import unittest

import torch
from torch import nn

from emperor.neuron._cluster.checkpointing import _NeuronClusterCheckpointingMixin


class _CheckpointNeuron(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(()))
        self.register_buffer("batch_counter", torch.zeros((), dtype=torch.int64))
        self.register_buffer("atrophy_counter", torch.zeros((), dtype=torch.int64))


class _CheckpointCluster(_NeuronClusterCheckpointingMixin, nn.Module):
    def __init__(
        self,
        neuron_names: tuple[str, ...] = ("neuron_1_1_1",),
        *,
        capacity: int = 2,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.cluster = nn.ModuleDict(
            {neuron_name: _CheckpointNeuron() for neuron_name in neuron_names}
        )
        self.register_buffer(
            "entry_coordinates",
            torch.tensor([[1, 1, 1]], dtype=torch.long),
            persistent=False,
        )
        self.forwards_since_last_growth = None
        self.total_growth_count = None
        self._checkpoint_removed_parameter_ids: set[int] = set()
        self.register_load_state_dict_pre_hook(self._reconcile_cluster_with_state_dict)

    @staticmethod
    def _is_neuron_name(neuron_name: str) -> bool:
        return re.fullmatch(r"neuron_-?\d+_-?\d+_-?\d+", neuron_name) is not None

    @staticmethod
    def _parse_neuron_name(neuron_name: str) -> tuple[int, int, int]:
        coordinates = neuron_name.removeprefix("neuron_").split("_")
        return tuple(int(value) for value in coordinates)

    @staticmethod
    def _neuron_name(x: int, y: int, z: int) -> str:
        return f"neuron_{x}_{y}_{z}"

    def _is_within_grid_capacity(self, position: tuple[int, int, int]) -> bool:
        x, y, z = position
        return 1 <= x <= self.capacity and y == 1 and z == 1

    @staticmethod
    def _coordinate_from_row(row: list[int]) -> tuple[int, int, int]:
        return tuple(row)

    @staticmethod
    def _initialize_neuron(_x: int, _y: int, _z: int) -> _CheckpointNeuron:
        return _CheckpointNeuron()

    @staticmethod
    def _add_neuron(
        cluster: nn.ModuleDict,
        neuron_name: str,
        neuron: nn.Module,
    ) -> None:
        cluster[neuron_name] = neuron


def _renamed_state_dict(
    model: nn.Module,
    source_name: str,
    target_name: str,
) -> dict[str, torch.Tensor]:
    return {
        key.replace(source_name, target_name): value.clone()
        for key, value in model.state_dict().items()
    }


class TestNeuronCheckpointTopologyValidation(unittest.TestCase):
    def test_invalid_topologies_are_rejected_before_mutation(self) -> None:
        cases = (
            (
                "neuron_2_1_1",
                1,
                "contains neurons outside the configured cluster capacity",
            ),
            (
                "neuron_2_1_1",
                2,
                "is missing configured entry-plane neurons",
            ),
            (
                "neuron_01_1_1",
                2,
                "contains non-canonical neuron names",
            ),
        )

        for incoming_name, capacity, expected_message in cases:
            with self.subTest(incoming_name=incoming_name):
                model = _CheckpointCluster(capacity=capacity)
                original_neuron = model.cluster["neuron_1_1_1"]
                incoming_state = _renamed_state_dict(
                    model,
                    "neuron_1_1_1",
                    incoming_name,
                )

                with self.assertRaisesRegex(RuntimeError, expected_message):
                    model.load_state_dict(incoming_state, strict=True)

                self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))
                self.assertIs(model.cluster["neuron_1_1_1"], original_neuron)


class TestNeuronCheckpointTopologyReconciliation(unittest.TestCase):
    def test_load_restores_saved_order_without_replacing_existing_neurons(self) -> None:
        source = _CheckpointCluster(("neuron_2_1_1", "neuron_1_1_1"), capacity=3)
        target = _CheckpointCluster(("neuron_1_1_1", "neuron_3_1_1"), capacity=3)
        existing_entry_neuron = target.cluster["neuron_1_1_1"]

        target.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(
            tuple(target.cluster),
            ("neuron_2_1_1", "neuron_1_1_1"),
        )
        self.assertIs(target.cluster["neuron_1_1_1"], existing_entry_neuron)

    def test_rebuilding_missing_neurons_preserves_the_rng_stream(self) -> None:
        source = _CheckpointCluster(("neuron_1_1_1", "neuron_2_1_1"))
        target = _CheckpointCluster()
        torch.manual_seed(20260720)
        expected_rng_state = torch.random.get_rng_state().clone()

        target.load_state_dict(source.state_dict(), strict=True)

        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng_state)
