import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from emperor.neuron._cluster.plasticity import ClusterPlasticityDelegate
from emperor.neuron._cluster.topology import ClusterTopologyDelegate

_COPY_AND_PERTURB_PARAMETERS = (
    "_ClusterPlasticityDelegate__copy_and_perturb_parent_parameters"
)
_SATURATED_NEURONS_BY_COUNTER = (
    "_ClusterPlasticityDelegate__saturated_neurons_by_descending_counter"
)


class _SingleParameterModule(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(values.clone())


class _TiedAndUntiedParameterModule(nn.Module):
    def __init__(
        self,
        shared_values: torch.Tensor,
        later_values: torch.Tensor,
    ) -> None:
        super().__init__()
        shared_parameter = nn.Parameter(shared_values.clone())
        self.shared_role_a = shared_parameter
        self.shared_role_b = shared_parameter
        self.later_weight = nn.Parameter(later_values.clone())


class _WarmupModule(nn.Module):
    def __init__(self, remaining_steps: int) -> None:
        super().__init__()
        self.register_buffer(
            "warmup_remaining_steps",
            torch.tensor(remaining_steps, dtype=torch.int64),
        )


class TestNeuronPlasticityNumerics(unittest.TestCase):
    def test_mitosis_uses_exact_float64_population_variance_scale(self) -> None:
        parent = _SingleParameterModule(
            torch.tensor(
                [100_000_000.0, 100_000_002.0] * 4,
                dtype=torch.float64,
            )
        )
        grown = _SingleParameterModule(torch.zeros(8, dtype=torch.float64))
        parent_before = parent.weight.detach().clone()
        population_std = parent_before.std(correction=0)
        torch.manual_seed(20260719)
        rng_state = torch.random.get_rng_state().clone()
        expected_noise = torch.randn_like(parent.weight)
        expected_weight = parent_before + expected_noise * population_std * 0.01
        torch.random.set_rng_state(rng_state)

        owner = SimpleNamespace()

        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        copy_and_perturb = (
            plasticity._ClusterPlasticityDelegate__copy_and_perturb_parent_parameters
        )
        result = copy_and_perturb(grown, parent)

        self.assertIs(result, grown)
        self.assertEqual(float(population_std), 1.0)
        torch.testing.assert_close(
            grown.weight,
            expected_weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            parent.weight,
            parent_before,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(grown.weight.requires_grad)

    def test_mitosis_tied_alias_does_not_skip_later_parameter_role(self) -> None:
        parent = _TiedAndUntiedParameterModule(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([10.0, 14.0], dtype=torch.float64),
        )
        grown = _TiedAndUntiedParameterModule(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        )
        parent_shared_before = parent.shared_role_a.detach().clone()
        parent_later_before = parent.later_weight.detach().clone()
        torch.manual_seed(20260720)
        rng_state = torch.random.get_rng_state().clone()
        shared_noise = torch.randn_like(parent.shared_role_a)
        later_noise = torch.randn_like(parent.later_weight)
        expected_shared = (
            parent_shared_before
            + shared_noise * parent_shared_before.std(correction=0) * 0.01
        )
        expected_later = (
            parent_later_before
            + later_noise * parent_later_before.std(correction=0) * 0.01
        )
        expected_rng_state = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng_state)

        owner = SimpleNamespace()

        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        copy_and_perturb = (
            plasticity._ClusterPlasticityDelegate__copy_and_perturb_parent_parameters
        )
        copy_and_perturb(grown, parent)

        self.assertIs(grown.shared_role_a, grown.shared_role_b)
        torch.testing.assert_close(
            grown.shared_role_a,
            expected_shared,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            grown.later_weight,
            expected_later,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(parent.shared_role_a, parent_shared_before)
        torch.testing.assert_close(parent.later_weight, parent_later_before)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng_state)

    def test_low_precision_mitosis_uses_population_scale_in_parameter_dtype(
        self,
    ) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                parent = _SingleParameterModule(
                    torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
                )
                grown = _SingleParameterModule(torch.zeros(3, dtype=dtype))
                population_std = torch.tensor(
                    math.sqrt(2.0 / 3.0),
                    dtype=torch.float32,
                ).to(dtype)
                torch.manual_seed(20260721)
                rng_state = torch.random.get_rng_state().clone()
                expected_noise = torch.randn_like(parent.weight)
                expected_rng_state = torch.random.get_rng_state().clone()
                expected_weight = (
                    parent.weight.detach() + expected_noise * population_std * 0.01
                )
                torch.random.set_rng_state(rng_state)

                owner = SimpleNamespace()

                plasticity = ClusterPlasticityDelegate(
                    owner, ClusterTopologyDelegate(owner)
                )
                copy_and_perturb = getattr(
                    plasticity,
                    _COPY_AND_PERTURB_PARAMETERS,
                )
                copy_and_perturb(grown, parent)

                self.assertTrue(torch.equal(grown.weight, expected_weight))
                self.assertEqual(grown.weight.dtype, dtype)
                torch.testing.assert_close(
                    torch.random.get_rng_state(),
                    expected_rng_state,
                )

    def test_complex_mitosis_preserves_population_scale_dtype_and_rng(self) -> None:
        for dtype in (torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                parent = _SingleParameterModule(
                    torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=dtype)
                )
                grown = _SingleParameterModule(torch.zeros(2, dtype=dtype))
                original_parent = parent.weight.detach().clone()
                population_std = original_parent.std(correction=0)
                torch.manual_seed(20260905)
                rng_state = torch.random.get_rng_state().clone()
                expected_noise = torch.randn_like(parent.weight)
                expected_weight = (
                    original_parent + expected_noise * population_std.to(dtype) * 0.01
                )
                expected_rng_state = torch.random.get_rng_state().clone()
                torch.random.set_rng_state(rng_state)
                owner = SimpleNamespace()
                plasticity = ClusterPlasticityDelegate(
                    owner, ClusterTopologyDelegate(owner)
                )

                result = getattr(plasticity, _COPY_AND_PERTURB_PARAMETERS)(
                    grown, parent
                )

                self.assertIs(result, grown)
                self.assertEqual(grown.weight.dtype, dtype)
                self.assertTrue(grown.weight.requires_grad)
                torch.testing.assert_close(
                    grown.weight, expected_weight, rtol=0.0, atol=0.0
                )
                torch.testing.assert_close(
                    parent.weight, original_parent, rtol=0.0, atol=0.0
                )
                torch.testing.assert_close(
                    torch.random.get_rng_state(), expected_rng_state
                )

    def test_mitosis_at_noise_floor_copies_without_consuming_rng(self) -> None:
        parent = _SingleParameterModule(
            torch.tensor([-1e-6, 1e-6], dtype=torch.float64)
        )
        grown = _SingleParameterModule(torch.zeros(2, dtype=torch.float64))
        torch.manual_seed(20260722)
        expected_rng_state = torch.random.get_rng_state().clone()

        owner = SimpleNamespace()

        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        copy_and_perturb = (
            plasticity._ClusterPlasticityDelegate__copy_and_perturb_parent_parameters
        )
        copy_and_perturb(grown, parent)

        torch.testing.assert_close(grown.weight, parent.weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng_state)


class TestNeuronPlasticityStateContracts(unittest.TestCase):
    def test_growth_priority_depends_on_counter_not_name_or_insertion_order(
        self,
    ) -> None:
        owner = SimpleNamespace()
        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        owner.growth_threshold = 2
        owner.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": nn.Identity(),
                "neuron_5_1_1": nn.Identity(),
                "neuron_9_1_1": nn.Identity(),
                "neuron_3_1_1": nn.Identity(),
            }
        )
        synchronized_counters = {
            "neuron_1_1_1": 8,
            "neuron_5_1_1": 3,
            "neuron_9_1_1": 5,
            "neuron_3_1_1": 1,
        }
        saturated_by_counter = getattr(
            plasticity,
            _SATURATED_NEURONS_BY_COUNTER,
        )

        ordered_names = [
            name for name, _ in saturated_by_counter(synchronized_counters)
        ]

        self.assertEqual(
            ordered_names,
            ["neuron_1_1_1", "neuron_9_1_1", "neuron_5_1_1"],
        )

    def test_growth_registers_and_initializes_one_neuron_before_parent_reset(
        self,
    ) -> None:
        def neuron_with_counter(counter, connection):
            neuron = nn.Module()
            neuron.register_buffer("batch_counter", torch.tensor(counter))
            neuron.terminal = SimpleNamespace(
                neuron_connections=torch.tensor([connection])
            )
            return neuron

        parent = neuron_with_counter(4, [2, 1, 1])
        other_parent = neuron_with_counter(3, [4, 1, 1])
        grown = neuron_with_counter(0, [1, 1, 1])
        owner = SimpleNamespace(
            cluster=nn.ModuleDict(
                {"neuron_1_1_1": parent, "neuron_3_1_1": other_parent}
            ),
            x_axis_total_neurons=4,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            growth_threshold=2,
            growth_cooldown_steps=3,
            forwards_since_last_growth=torch.tensor(2),
            max_total_growths=2,
            total_growth_count=torch.tensor(0),
            growth_warmup_steps=3,
            mitosis_initialization_flag=False,
            escape_counts=torch.tensor([0, 7, 0, 9]).reshape(4, 1, 1),
        )
        events = []

        def initialize_neuron(x, y, z, *, runtime_template):
            events.append(
                (
                    "initialize",
                    (x, y, z),
                    runtime_template is parent,
                    int(parent.batch_counter),
                    int(owner.forwards_since_last_growth),
                )
            )
            return grown

        def add_neuron(cluster, name, neuron):
            events.append(("register", name, hasattr(neuron, "warmup_remaining_steps")))
            cluster[name] = neuron

        class CalledNeuronNames(set):
            def add(self, name):
                events.append(
                    (
                        "called",
                        name,
                        int(grown.warmup_remaining_steps),
                        int(owner.escape_counts[1, 0, 0]),
                        int(owner.total_growth_count),
                        int(owner.forwards_since_last_growth),
                        int(parent.batch_counter),
                    )
                )
                super().add(name)

        owner._initialize_neuron = initialize_neuron
        owner._add_neuron = add_neuron
        forward_context = SimpleNamespace(called_neuron_names=CalledNeuronNames())
        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))

        plasticity.check_neuron_growth(None, forward_context)

        self.assertEqual(
            events,
            [
                ("initialize", (2, 1, 1), True, 4, 3),
                ("register", "neuron_2_1_1", False),
                ("called", "neuron_2_1_1", 3, 0, 1, 0, 4),
            ],
        )
        self.assertEqual(
            tuple(owner.cluster),
            ("neuron_1_1_1", "neuron_3_1_1", "neuron_2_1_1"),
        )
        self.assertIs(owner.cluster["neuron_2_1_1"], grown)
        self.assertEqual(forward_context.called_neuron_names, {"neuron_2_1_1"})
        self.assertEqual(int(parent.batch_counter), 0)
        self.assertEqual(int(other_parent.batch_counter), 3)
        self.assertEqual(int(owner.escape_counts[3, 0, 0]), 9)

    def test_non_cubic_escape_counter_mapping_preserves_all_three_axes(self) -> None:
        owner = SimpleNamespace()
        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        owner.training = True
        owner.y_axis_total_neurons = 3
        owner.z_axis_total_neurons = 4
        owner.escape_counts = torch.zeros((2, 3, 4), dtype=torch.int64)
        positions = [(1, 2, 3), (2, 1, 4), (2, 3, 2), (1, 2, 3)]

        plasticity.record_escaped_missing_positions(positions)

        expected_counts = torch.zeros_like(owner.escape_counts)
        expected_counts[0, 1, 2] = 2
        expected_counts[1, 0, 3] = 1
        expected_counts[1, 2, 1] = 1
        torch.testing.assert_close(owner.escape_counts, expected_counts)

    def test_warmup_countdown_reaches_zero_and_does_not_underflow(self) -> None:
        owner = SimpleNamespace()
        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        owner.growth_warmup_steps = 2
        warmup_module = _WarmupModule(2)
        owner.cluster = nn.ModuleDict({"grown": warmup_module})

        observed_countdown = []
        for _ in range(3):
            plasticity.advance_grown_neuron_warmup()
            observed_countdown.append(int(warmup_module.warmup_remaining_steps))

        self.assertEqual(observed_countdown, [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
