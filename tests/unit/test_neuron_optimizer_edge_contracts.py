import copy
import gc
import pickle
import unittest
import weakref
from types import SimpleNamespace

import torch
import torch.nn as nn
from lightning.pytorch.trainer.states import TrainerFn

from emperor.neuron import NeuronClusterOptimizerSyncCallback
from emperor.neuron._optimizer_checkpoint import (
    LegacyOptimizerAppendPolicy,
    NeuronOptimizerCheckpointReconciler,
    NeuronOptimizerLoadTransaction,
)
from emperor.neuron._optimizer_layout import (
    OPTIMIZER_LAYOUT_CHECKPOINT_KEY,
    NeuronOptimizerNamedLayout,
)
from emperor.neuron._optimizer_scheduler import (
    NeuronSchedulerCheckpointReconciler,
    SchedulerGroupLoadBinding,
    extend_scheduler_for_new_group,
    preflight_scheduler_group_extension,
    preflight_scheduler_group_removal,
    reconcile_scheduler_group_count,
    remove_scheduler_groups,
)


class _DynamicCluster(nn.Module):
    def __init__(self, *, include_dynamic_neuron: bool = True) -> None:
        super().__init__()
        neurons = {"neuron_0_0_0": nn.Linear(1, 1)}
        if include_dynamic_neuron:
            neurons["neuron_1_0_0"] = nn.Linear(1, 1)
        self.cluster = nn.ModuleDict(neurons)
        self.initial_x_axis_start = 0
        self.initial_x_axis_total_neurons = 1
        self.initial_y_axis_start = 0
        self.initial_y_axis_total_neurons = 1
        self.initial_z_axis_start = 0
        self.initial_z_axis_total_neurons = 1

    @staticmethod
    def _neuron_name(x_coordinate: int, y_coordinate: int, z_coordinate: int) -> str:
        return f"neuron_{x_coordinate}_{y_coordinate}_{z_coordinate}"


class _ExplodingList(list):
    def extend(self, values) -> None:
        raise RuntimeError("injected scheduler mutation failure")


class _ExplodingOnceDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._should_explode = True

    def __setitem__(self, name, value) -> None:
        if name == "params" and self._should_explode:
            self._should_explode = False
            raise RuntimeError("injected named-layout migration failure")
        super().__setitem__(name, value)


class _FailingAddParamGroupSGD(torch.optim.SGD):
    fail_next_add = False

    def add_param_group(self, param_group) -> None:
        if self.fail_next_add:
            self.fail_next_add = False
            raise RuntimeError("injected first add_param_group failure")
        super().add_param_group(param_group)


class TestNeuronOptimizerCheckpointEdges(unittest.TestCase):
    @staticmethod
    def legacy_fixture(
        *,
        include_names: bool = False,
        include_nested_options: bool = False,
    ):
        cluster = _DynamicCluster()
        optimizer = torch.optim.SGD(cluster.parameters(), lr=0.1, momentum=0.9)
        if include_nested_options:
            optimizer.param_groups[0]["tensor_option"] = torch.tensor(2.0)
            optimizer.param_groups[0]["nested_option"] = {
                "scales": [torch.tensor(3.0), 4.0]
            }
        names_by_parameter_id = {
            id(parameter): name for name, parameter in cluster.named_parameters()
        }
        if include_names:
            optimizer.param_groups[0]["param_names"] = [
                names_by_parameter_id[id(parameter)]
                for parameter in optimizer.param_groups[0]["params"]
            ]
        serialized_state = optimizer.state_dict()
        serialized_ids_by_parameter_id = {
            id(parameter): serialized_id
            for parameter, serialized_id in zip(
                optimizer.param_groups[0]["params"],
                serialized_state["param_groups"][0]["params"],
                strict=True,
            )
        }
        dynamic_parameter_ids = {
            id(parameter) for parameter in cluster.cluster["neuron_1_0_0"].parameters()
        }
        base_parameters = [
            parameter
            for parameter in optimizer.param_groups[0]["params"]
            if id(parameter) not in dynamic_parameter_ids
        ]
        dynamic_parameters = list(cluster.cluster["neuron_1_0_0"].parameters())
        saved_options = {
            name: value
            for name, value in serialized_state["param_groups"][0].items()
            if name not in {"params", "param_names"}
        }

        def saved_group(parameters):
            group = {
                **saved_options,
                "params": [
                    serialized_ids_by_parameter_id[id(parameter)]
                    for parameter in parameters
                ],
            }
            if include_names:
                group["param_names"] = [
                    names_by_parameter_id[id(parameter)] for parameter in parameters
                ]
            return group

        saved_state = {
            "state": serialized_state["state"],
            "param_groups": [
                saved_group(base_parameters),
                saved_group(dynamic_parameters),
            ],
        }
        return cluster, optimizer, saved_state

    def test_load_transaction_requires_every_optimizer_and_ignores_strangers(
        self,
    ) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(3)]
        optimizers = [torch.optim.SGD([parameter], lr=0.1) for parameter in parameters]
        transaction = NeuronOptimizerLoadTransaction()

        self.assertFalse(transaction.optimizer_requires_completion(optimizers[0]))
        transaction.prepare_for_load(optimizers[:2])
        transaction.mark_optimizer_loaded(optimizers[2])
        self.assertTrue(transaction.optimizer_requires_completion(optimizers[0]))
        transaction.mark_optimizer_loaded(optimizers[0])

        with self.assertRaisesRegex(RuntimeError, "partial Neuron optimizer"):
            transaction.commit_loaded()

        transaction.mark_optimizer_loaded(optimizers[1])
        transaction.commit_loaded()
        self.assertFalse(transaction.optimizer_requires_completion(optimizers[0]))

    def test_legacy_reconciler_ignores_nonlegacy_payload_shapes(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [], [])
        reconciler.prepare_for_load([optimizer], [], [{"state": {}}])
        reconciler.prepare_for_load(
            [optimizer],
            [],
            [{"state": {}, "param_groups": optimizer.state_dict()["param_groups"]}],
        )

        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rejects_ambiguous_or_mismatched_payloads(
        self,
    ) -> None:
        reconciler = NeuronOptimizerCheckpointReconciler()
        dynamic_cluster = _DynamicCluster()
        external_optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=0.1)
        external_state = external_optimizer.state_dict()
        external_state["param_groups"].append(
            copy.deepcopy(external_state["param_groups"][0])
        )

        with self.assertRaisesRegex(RuntimeError, "exactly one optimizer-owned"):
            reconciler.prepare_for_load(
                [external_optimizer],
                [dynamic_cluster],
                [external_state],
            )

        static_cluster = _DynamicCluster(include_dynamic_neuron=False)
        static_optimizer = torch.optim.SGD(static_cluster.parameters(), lr=0.1)
        static_state = static_optimizer.state_dict()
        static_state["param_groups"].append(
            copy.deepcopy(static_state["param_groups"][0])
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one optimizer-owned"):
            reconciler.prepare_for_load(
                [static_optimizer],
                [static_cluster],
                [static_state],
            )

        cluster, optimizer, saved_state = self.legacy_fixture()
        saved_state["param_groups"][0]["params"].pop()
        with self.assertRaisesRegex(RuntimeError, "base parameter-group sizes"):
            reconciler.prepare_for_load([optimizer], [cluster], [saved_state])

        cluster, optimizer, saved_state = self.legacy_fixture()
        saved_state["param_groups"][1]["params"].pop()
        with self.assertRaisesRegex(RuntimeError, "appended parameter count"):
            reconciler.prepare_for_load([optimizer], [cluster], [saved_state])

        cluster, optimizer, saved_state = self.legacy_fixture()
        saved_state["param_groups"][1].pop("momentum")
        with self.assertRaisesRegex(RuntimeError, "optimizer options diverge"):
            reconciler.prepare_for_load([optimizer], [cluster], [saved_state])

    def test_legacy_reconciler_compares_nested_options_and_restores_names(
        self,
    ) -> None:
        cluster, optimizer, saved_state = self.legacy_fixture(
            include_names=True,
            include_nested_options=True,
        )
        original_parameters = list(optimizer.param_groups[0]["params"])
        original_names = list(optimizer.param_groups[0]["param_names"])
        expected_group_parameters = [
            list(cluster.cluster[neuron_name].parameters())
            for neuron_name in ("neuron_0_0_0", "neuron_1_0_0")
        ]
        parameter_names_by_id = {
            id(parameter): name for name, parameter in cluster.named_parameters()
        }
        expected_group_parameter_names = [
            [parameter_names_by_id[id(parameter)] for parameter in parameters]
            for parameters in expected_group_parameters
        ]
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [saved_state])
        self.assertEqual(len(optimizer.param_groups), 2)
        for group in optimizer.param_groups:
            self.assertIsInstance(group["params"], list)
            self.assertIsInstance(group["param_names"], list)
            self.assertEqual(len(group["params"]), len(group["param_names"]))
        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in parameters]
                for parameters in expected_group_parameters
            ],
        )
        self.assertEqual(
            [group["param_names"] for group in optimizer.param_groups],
            expected_group_parameter_names,
        )
        reconciler.clear()

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(
            [id(parameter) for parameter in optimizer.param_groups[0]["params"]],
            [id(parameter) for parameter in original_parameters],
        )
        self.assertEqual(optimizer.param_groups[0]["param_names"], original_names)

    def test_legacy_reconciler_maps_three_suffix_groups_cumulatively(self) -> None:
        cluster = _DynamicCluster()
        cluster.cluster["neuron_2_0_0"] = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(cluster.parameters(), lr=0.1, momentum=0.9)
        all_parameters = list(cluster.parameters())
        for sentinel, parameter in enumerate(all_parameters, start=11):
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
        dynamic_parameters = [
            *cluster.cluster["neuron_1_0_0"].parameters(),
            *cluster.cluster["neuron_2_0_0"].parameters(),
        ]
        expected_suffixes = [
            dynamic_parameters[:1],
            dynamic_parameters[1:2],
            dynamic_parameters[2:],
        ]
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

        saved_state = {
            "state": serialized_state["state"],
            "param_groups": [
                saved_group(base_parameters),
                *(saved_group(parameters) for parameters in expected_suffixes),
            ],
        }
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [saved_state])
        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups[1:]
            ],
            [
                [id(parameter) for parameter in parameters]
                for parameters in expected_suffixes
            ],
        )
        optimizer.load_state_dict(saved_state)
        self.assertEqual(
            reconciler.complete_optimizer_load(optimizer),
            LegacyOptimizerAppendPolicy(1, 0, (0, 0, 0, 0)),
        )

        for sentinel, parameter in enumerate(all_parameters, start=11):
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                torch.full_like(parameter, float(sentinel)),
            )

    def test_legacy_reconciler_first_group_add_failure_is_exactly_atomic(
        self,
    ) -> None:
        cluster, _, saved_state = self.legacy_fixture(include_names=True)
        optimizer = _FailingAddParamGroupSGD(
            cluster.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        names_by_parameter_id = {
            id(parameter): name for name, parameter in cluster.named_parameters()
        }
        optimizer.param_groups[0]["param_names"] = [
            names_by_parameter_id[id(parameter)]
            for parameter in optimizer.param_groups[0]["params"]
        ]
        state_parameter = optimizer.param_groups[0]["params"][0]
        state_sentinel = {"momentum_buffer": torch.full_like(state_parameter, 17.0)}
        optimizer.state[state_parameter] = state_sentinel
        original_group_list = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        original_parameters = tuple(original_parameter_list)
        original_name_list = original_group["param_names"]
        original_names = tuple(original_name_list)
        original_state = optimizer.state
        original_state_items = tuple(original_state.items())
        reconciler = NeuronOptimizerCheckpointReconciler()
        optimizer.fail_next_add = True

        with self.assertRaisesRegex(
            RuntimeError,
            "injected first add_param_group failure",
        ):
            reconciler.prepare_for_load([optimizer], [cluster], [saved_state])

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIs(optimizer.param_groups[0], original_group)
        self.assertIs(original_group["params"], original_parameter_list)
        self.assertEqual(tuple(original_parameter_list), original_parameters)
        self.assertIs(original_group["param_names"], original_name_list)
        self.assertEqual(tuple(original_name_list), original_names)
        self.assertIs(optimizer.state, original_state)
        self.assertEqual(tuple(optimizer.state.items()), original_state_items)
        self.assertIs(optimizer.state[state_parameter], state_sentinel)
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rejects_nested_option_divergence_atomically(
        self,
    ) -> None:
        divergent_options = (
            ("unequal-tensors", "tensor_option", torch.tensor(9.0)),
            ("tensor-versus-scalar", "tensor_option", 2.0),
            ("dict-versus-scalar", "nested_option", 7.0),
            (
                "equal-length-sequence-difference",
                "nested_option",
                {"scales": [torch.tensor(3.0), 5.0]},
            ),
        )

        for case_name, option_name, divergent_value in divergent_options:
            with self.subTest(case=case_name):
                cluster, optimizer, saved_state = self.legacy_fixture(
                    include_names=True,
                    include_nested_options=True,
                )
                saved_state["param_groups"][1][option_name] = divergent_value
                state_parameter = optimizer.param_groups[0]["params"][0]
                state_sentinel = {
                    "momentum_buffer": torch.full_like(state_parameter, 13.0)
                }
                optimizer.state[state_parameter] = state_sentinel
                original_param_groups = optimizer.param_groups
                original_optimizer_state = optimizer.state
                original_optimizer_state_items = tuple(optimizer.state.items())
                original_saved_state = saved_state["state"]

                def capture_mapping_snapshots(mappings):
                    return tuple(
                        (
                            mapping,
                            dict(mapping),
                            {
                                name: (value, tuple(value))
                                for name, value in mapping.items()
                                if isinstance(value, list)
                            },
                        )
                        for mapping in mappings
                    )

                optimizer_group_snapshots = capture_mapping_snapshots(
                    optimizer.param_groups
                )
                saved_group_list = saved_state["param_groups"]
                saved_group_snapshots = capture_mapping_snapshots(saved_group_list)
                reconciler = NeuronOptimizerCheckpointReconciler()

                with self.assertRaisesRegex(
                    RuntimeError,
                    "optimizer options diverge",
                ):
                    reconciler.prepare_for_load(
                        [optimizer],
                        [cluster],
                        [saved_state],
                    )

                self.assertIs(optimizer.param_groups, original_param_groups)
                self.assertIs(optimizer.state, original_optimizer_state)
                self.assertEqual(
                    tuple(optimizer.state.items()),
                    original_optimizer_state_items,
                )
                self.assertIs(optimizer.state[state_parameter], state_sentinel)
                self.assertIs(saved_state["param_groups"], saved_group_list)
                self.assertIs(saved_state["state"], original_saved_state)
                for mapping_snapshots in (
                    optimizer_group_snapshots,
                    saved_group_snapshots,
                ):
                    for mapping, original_values, original_lists in mapping_snapshots:
                        self.assertEqual(set(mapping), set(original_values))
                        for name, original_value in original_values.items():
                            if name not in original_lists:
                                self.assertIs(mapping[name], original_value)
                                continue
                            original_list, original_items = original_lists[name]
                            self.assertIs(mapping[name], original_list)
                            self.assertEqual(len(mapping[name]), len(original_items))
                            for value, original_item in zip(
                                mapping[name],
                                original_items,
                                strict=True,
                            ):
                                self.assertIs(value, original_item)
                self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rolls_back_earlier_optimizer_on_later_failure(
        self,
    ) -> None:
        first_cluster, first_optimizer, first_state = self.legacy_fixture(
            include_names=True,
            include_nested_options=True,
        )
        second_cluster, second_optimizer, second_state = self.legacy_fixture(
            include_names=True,
            include_nested_options=True,
        )
        second_state["param_groups"][1]["tensor_option"] = torch.tensor(19.0)
        original_group_lists = (
            first_optimizer.param_groups,
            second_optimizer.param_groups,
        )
        original_groups = tuple(
            optimizer.param_groups[0]
            for optimizer in (first_optimizer, second_optimizer)
        )
        original_parameter_lists = tuple(group["params"] for group in original_groups)
        original_parameters = tuple(tuple(group["params"]) for group in original_groups)
        original_name_lists = tuple(group["param_names"] for group in original_groups)
        original_names = tuple(tuple(group["param_names"]) for group in original_groups)
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "optimizer options diverge"):
            reconciler.prepare_for_load(
                [first_optimizer, second_optimizer],
                [first_cluster, second_cluster],
                [first_state, second_state],
            )

        for index, optimizer in enumerate((first_optimizer, second_optimizer)):
            self.assertIs(optimizer.param_groups, original_group_lists[index])
            self.assertEqual(len(optimizer.param_groups), 1)
            self.assertIs(optimizer.param_groups[0], original_groups[index])
            self.assertIs(
                optimizer.param_groups[0]["params"],
                original_parameter_lists[index],
            )
            self.assertEqual(
                tuple(optimizer.param_groups[0]["params"]),
                original_parameters[index],
            )
            self.assertIs(
                optimizer.param_groups[0]["param_names"],
                original_name_lists[index],
            )
            self.assertEqual(
                tuple(optimizer.param_groups[0]["param_names"]),
                original_names[index],
            )
            self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

        second_state["param_groups"][1]["tensor_option"] = second_state["param_groups"][
            0
        ]["tensor_option"]
        reconciler.prepare_for_load(
            [first_optimizer, second_optimizer],
            [first_cluster, second_cluster],
            [first_state, second_state],
        )
        self.assertEqual(len(first_optimizer.param_groups), 2)
        self.assertEqual(len(second_optimizer.param_groups), 2)
        reconciler.clear()
        self.assertEqual(len(first_optimizer.param_groups), 1)
        self.assertEqual(len(second_optimizer.param_groups), 1)

    def test_legacy_reconciler_maps_frozen_dynamic_parameters_by_owned_role(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        cluster.requires_grad_(False)

        def optimizer_and_legacy_state(initial_parameter, dynamic_parameter):
            optimizer = torch.optim.SGD(
                [initial_parameter, dynamic_parameter],
                lr=0.1,
                momentum=0.9,
            )
            serialized_group = optimizer.state_dict()["param_groups"][0]
            group_options = {
                name: value
                for name, value in serialized_group.items()
                if name != "params"
            }
            return optimizer, {
                "state": {},
                "param_groups": [
                    {**group_options, "params": [0]},
                    {**group_options, "params": [1]},
                ],
            }

        weight_optimizer, weight_state = optimizer_and_legacy_state(
            initial_neuron.weight,
            dynamic_neuron.weight,
        )
        bias_optimizer, bias_state = optimizer_and_legacy_state(
            initial_neuron.bias,
            dynamic_neuron.bias,
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [weight_optimizer, bias_optimizer],
            [cluster],
            [weight_state, bias_state],
        )
        weight_optimizer.load_state_dict(weight_state)
        bias_optimizer.load_state_dict(bias_state)

        self.assertTrue(reconciler.complete_optimizer_load(weight_optimizer))
        self.assertTrue(reconciler.complete_optimizer_load(bias_optimizer))
        self.assertIs(
            weight_optimizer.param_groups[1]["params"][0], dynamic_neuron.weight
        )
        self.assertIs(bias_optimizer.param_groups[1]["params"][0], dynamic_neuron.bias)
        self.assertNotIn(
            id(dynamic_neuron.bias),
            {id(parameter) for parameter in weight_optimizer.param_groups[1]["params"]},
        )
        self.assertNotIn(
            id(dynamic_neuron.weight),
            {id(parameter) for parameter in bias_optimizer.param_groups[1]["params"]},
        )

    def test_legacy_reconciler_keeps_frozen_initial_dynamic_alias_in_base_group(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        dynamic_neuron.weight = initial_neuron.weight
        named_parameters = list(cluster.named_parameters())
        names_by_parameter_id = {
            id(parameter): name for name, parameter in named_parameters
        }
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [parameter for _, parameter in named_parameters],
                    "param_names": [name for name, _ in named_parameters],
                }
            ],
            lr=0.1,
            momentum=0.9,
        )
        expected_momentum_by_parameter_id = {}
        for index, parameter in enumerate(optimizer.param_groups[0]["params"], start=1):
            momentum = torch.full_like(parameter, float(index))
            optimizer.state[parameter]["momentum_buffer"] = momentum.clone()
            expected_momentum_by_parameter_id[id(parameter)] = momentum
        serialized_state = optimizer.state_dict()
        serialized_id_by_parameter_id = {
            id(parameter): serialized_id
            for parameter, serialized_id in zip(
                optimizer.param_groups[0]["params"],
                serialized_state["param_groups"][0]["params"],
                strict=True,
            )
        }
        group_options = {
            name: value
            for name, value in serialized_state["param_groups"][0].items()
            if name not in {"params", "param_names"}
        }

        def saved_group(parameters):
            return {
                **group_options,
                "params": [
                    serialized_id_by_parameter_id[id(parameter)]
                    for parameter in parameters
                ],
                "param_names": [
                    names_by_parameter_id[id(parameter)] for parameter in parameters
                ],
            }

        base_parameters = [initial_neuron.weight, initial_neuron.bias]
        dynamic_parameters = [dynamic_neuron.bias]
        legacy_state = {
            "state": serialized_state["state"],
            "param_groups": [
                saved_group(base_parameters),
                saved_group(dynamic_parameters),
            ],
        }
        cluster.requires_grad_(False)
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [legacy_state])
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in base_parameters],
                [id(parameter) for parameter in dynamic_parameters],
            ],
        )
        self.assertEqual(
            [name for group in optimizer.param_groups for name in group["param_names"]],
            [names_by_parameter_id[id(parameter)] for _, parameter in named_parameters],
        )
        optimizer_parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(
            len(optimizer_parameter_ids), len(set(optimizer_parameter_ids))
        )
        for _, parameter in named_parameters:
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                expected_momentum_by_parameter_id[id(parameter)],
            )

        cluster.requires_grad_(True)
        before_shared_weight = initial_neuron.weight.detach().clone()
        initial_neuron.weight.grad = torch.ones_like(initial_neuron.weight)
        optimizer.step()

        self.assertFalse(torch.equal(initial_neuron.weight, before_shared_weight))

    def test_legacy_reconciler_keeps_cross_cluster_initial_alias_in_base_group(
        self,
    ) -> None:
        first_cluster = _DynamicCluster(include_dynamic_neuron=False)
        second_cluster = _DynamicCluster()
        first_initial_neuron = first_cluster.cluster["neuron_0_0_0"]
        second_initial_neuron = second_cluster.cluster["neuron_0_0_0"]
        second_dynamic_neuron = second_cluster.cluster["neuron_1_0_0"]
        second_dynamic_neuron.weight = first_initial_neuron.weight
        unique_parameters = []
        parameter_ids: set[int] = set()
        for parameter in (*first_cluster.parameters(), *second_cluster.parameters()):
            if id(parameter) in parameter_ids:
                continue
            unique_parameters.append(parameter)
            parameter_ids.add(id(parameter))
        optimizer = torch.optim.SGD(unique_parameters, lr=0.1, momentum=0.9)
        serialized_state = optimizer.state_dict()
        serialized_id_by_parameter_id = {
            id(parameter): serialized_id
            for parameter, serialized_id in zip(
                optimizer.param_groups[0]["params"],
                serialized_state["param_groups"][0]["params"],
                strict=True,
            )
        }
        group_options = {
            name: value
            for name, value in serialized_state["param_groups"][0].items()
            if name != "params"
        }
        base_parameters = [
            first_initial_neuron.weight,
            first_initial_neuron.bias,
            second_initial_neuron.weight,
            second_initial_neuron.bias,
        ]
        dynamic_parameters = [second_dynamic_neuron.bias]
        legacy_state = {
            "state": serialized_state["state"],
            "param_groups": [
                {
                    **group_options,
                    "params": [
                        serialized_id_by_parameter_id[id(parameter)]
                        for parameter in base_parameters
                    ],
                },
                {
                    **group_options,
                    "params": [
                        serialized_id_by_parameter_id[id(parameter)]
                        for parameter in dynamic_parameters
                    ],
                },
            ],
        }
        first_cluster.requires_grad_(False)
        second_cluster.requires_grad_(False)
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [optimizer],
            [first_cluster, second_cluster],
            [legacy_state],
        )
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in base_parameters],
                [id(parameter) for parameter in dynamic_parameters],
            ],
        )
        optimized_parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(
            optimized_parameter_ids.count(id(first_initial_neuron.weight)),
            1,
        )

    def test_legacy_reconciler_uses_global_initial_aliases_after_dynamic_cluster(
        self,
    ) -> None:
        dynamic_cluster = _DynamicCluster()
        alias_cluster = _DynamicCluster()
        dynamic_initial_neuron = dynamic_cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = dynamic_cluster.cluster["neuron_1_0_0"]
        alias_initial_neuron = alias_cluster.cluster["neuron_0_0_0"]
        alias_dynamic_neuron = alias_cluster.cluster["neuron_1_0_0"]
        alias_dynamic_neuron.weight = alias_initial_neuron.weight
        alias_dynamic_neuron.bias = alias_initial_neuron.bias
        unique_parameters = []
        parameter_ids: set[int] = set()
        for parameter in (*dynamic_cluster.parameters(), *alias_cluster.parameters()):
            if id(parameter) in parameter_ids:
                continue
            unique_parameters.append(parameter)
            parameter_ids.add(id(parameter))
        optimizer = torch.optim.SGD(unique_parameters, lr=0.1, momentum=0.9)
        serialized_state = optimizer.state_dict()
        serialized_id_by_parameter_id = {
            id(parameter): serialized_id
            for parameter, serialized_id in zip(
                optimizer.param_groups[0]["params"],
                serialized_state["param_groups"][0]["params"],
                strict=True,
            )
        }
        group_options = {
            name: value
            for name, value in serialized_state["param_groups"][0].items()
            if name != "params"
        }
        base_parameters = [
            dynamic_initial_neuron.weight,
            dynamic_initial_neuron.bias,
            alias_initial_neuron.weight,
            alias_initial_neuron.bias,
        ]
        dynamic_parameters = [dynamic_neuron.weight, dynamic_neuron.bias]
        legacy_state = {
            "state": serialized_state["state"],
            "param_groups": [
                {
                    **group_options,
                    "params": [
                        serialized_id_by_parameter_id[id(parameter)]
                        for parameter in base_parameters
                    ],
                },
                {
                    **group_options,
                    "params": [
                        serialized_id_by_parameter_id[id(parameter)]
                        for parameter in dynamic_parameters
                    ],
                },
            ],
        }
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [optimizer],
            [dynamic_cluster, alias_cluster],
            [legacy_state],
        )
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in base_parameters],
                [id(parameter) for parameter in dynamic_parameters],
            ],
        )

    def test_legacy_reconciler_rejects_unnamed_custom_external_alias_order(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        root_module.other = nn.Linear(1, 1)
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        dynamic_neuron.weight = root_module.other.weight
        optimizer = torch.optim.SGD(
            reversed(list(root_module.parameters())),
            lr=0.1,
            momentum=0.9,
        )
        serialized_state = optimizer.state_dict()
        serialized_id_by_parameter_id = {
            id(parameter): serialized_id
            for parameter, serialized_id in zip(
                optimizer.param_groups[0]["params"],
                serialized_state["param_groups"][0]["params"],
                strict=True,
            )
        }
        group_options = {
            name: value
            for name, value in serialized_state["param_groups"][0].items()
            if name != "params"
        }
        base_parameters = [
            initial_neuron.weight,
            initial_neuron.bias,
            root_module.other.weight,
            root_module.other.bias,
        ]
        legacy_state = {
            "state": serialized_state["state"],
            "param_groups": [
                {
                    **group_options,
                    "params": [
                        serialized_id_by_parameter_id[id(parameter)]
                        for parameter in base_parameters
                    ],
                },
                {
                    **group_options,
                    "params": [serialized_id_by_parameter_id[id(dynamic_neuron.bias)]],
                },
            ],
        }
        original_group_list = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        original_parameters = tuple(original_parameter_list)
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "custom optimizer parameter order"):
            reconciler.prepare_for_load(
                [optimizer],
                [cluster],
                [legacy_state],
                root_module=root_module,
            )

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIs(optimizer.param_groups[0], original_group)
        self.assertIs(optimizer.param_groups[0]["params"], original_parameter_list)
        self.assertEqual(
            tuple(optimizer.param_groups[0]["params"]), original_parameters
        )
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rejects_duplicate_base_alias_group_atomically(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        dynamic_neuron.weight = initial_neuron.weight
        optimizer = torch.optim.SGD(
            [initial_neuron.weight, initial_neuron.bias, dynamic_neuron.bias],
            lr=0.1,
            momentum=0.9,
        )
        optimizer.param_groups[0]["params"].insert(1, initial_neuron.weight)
        serialized_state = optimizer.state_dict()
        serialized_group = serialized_state["param_groups"][0]
        group_options = {
            name: value for name, value in serialized_group.items() if name != "params"
        }
        legacy_state = {
            "state": serialized_state["state"],
            "param_groups": [
                {**group_options, "params": serialized_group["params"][:-1]},
                {**group_options, "params": serialized_group["params"][-1:]},
            ],
        }
        original_group_list = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        original_parameters = tuple(original_parameter_list)
        original_state = optimizer.state
        original_state_items = tuple(optimizer.state.items())
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "duplicate parameters"):
            reconciler.prepare_for_load(
                [optimizer],
                [cluster],
                [legacy_state],
                root_module=root_module,
            )

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(tuple(optimizer.param_groups), (original_group,))
        self.assertIs(optimizer.param_groups[0]["params"], original_parameter_list)
        self.assertEqual(tuple(original_parameter_list), original_parameters)
        self.assertIs(optimizer.state, original_state)
        self.assertEqual(tuple(optimizer.state.items()), original_state_items)
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rejects_base_alias_group_outside_root_atomically(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        dynamic_neuron.weight = initial_neuron.weight
        foreign_parameter = nn.Parameter(torch.tensor([9.0]))
        optimizer = torch.optim.SGD(
            [
                initial_neuron.weight,
                initial_neuron.bias,
                foreign_parameter,
                dynamic_neuron.bias,
            ],
            lr=0.1,
            momentum=0.9,
        )
        serialized_state = optimizer.state_dict()
        serialized_group = serialized_state["param_groups"][0]
        group_options = {
            name: value for name, value in serialized_group.items() if name != "params"
        }
        legacy_state = {
            "state": serialized_state["state"],
            "param_groups": [
                {**group_options, "params": serialized_group["params"][:-1]},
                {**group_options, "params": serialized_group["params"][-1:]},
            ],
        }
        original_group_list = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        original_parameters = tuple(original_parameter_list)
        original_state = optimizer.state
        original_state_items = tuple(optimizer.state.items())
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "cannot reconstruct"):
            reconciler.prepare_for_load(
                [optimizer],
                [cluster],
                [legacy_state],
                root_module=root_module,
            )

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(tuple(optimizer.param_groups), (original_group,))
        self.assertIs(optimizer.param_groups[0]["params"], original_parameter_list)
        self.assertEqual(tuple(original_parameter_list), original_parameters)
        self.assertIs(optimizer.state, original_state)
        self.assertEqual(tuple(optimizer.state.items()), original_state_items)
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rejects_ambiguous_multigroup_external_aliases(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        root_module.external = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([[10.0]])),
                nn.Parameter(torch.tensor([20.0])),
            ]
        )
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        initial_neuron.register_parameter(
            "tail",
            nn.Parameter(torch.tensor([30.0])),
        )
        dynamic_neuron.weight = root_module.external[0]
        dynamic_neuron.bias = root_module.external[1]
        dynamic_neuron.register_parameter(
            "tail",
            nn.Parameter(torch.tensor([40.0])),
        )
        first_external_parameter, second_external_parameter = root_module.external
        dynamic_tail = dynamic_neuron.tail

        source_optimizer = torch.optim.SGD(
            [
                {
                    "params": [
                        initial_neuron.weight,
                        initial_neuron.tail,
                        first_external_parameter,
                    ],
                    "lr": 0.1,
                },
                {
                    "params": [initial_neuron.bias, second_external_parameter],
                    "lr": 0.2,
                },
                {"params": [dynamic_tail], "lr": 0.1},
            ],
            momentum=0.9,
        )
        for parameter, momentum_value in (
            (first_external_parameter, 11.0),
            (second_external_parameter, 22.0),
            (dynamic_tail, 33.0),
        ):
            source_optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                parameter,
                momentum_value,
            )
        legacy_state = source_optimizer.state_dict()

        optimizer = torch.optim.SGD(
            [
                {
                    "params": [
                        initial_neuron.weight,
                        initial_neuron.tail,
                        second_external_parameter,
                        dynamic_tail,
                    ],
                    "lr": 0.1,
                },
                {
                    "params": [initial_neuron.bias, first_external_parameter],
                    "lr": 0.2,
                },
            ],
            momentum=0.9,
        )
        original_group_list = optimizer.param_groups
        original_groups = tuple(optimizer.param_groups)
        original_parameter_lists = tuple(
            group["params"] for group in optimizer.param_groups
        )
        original_parameters = tuple(
            tuple(parameters) for parameters in original_parameter_lists
        )
        original_state = optimizer.state
        original_state_items = tuple(optimizer.state.items())
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "multiple base parameter groups"):
            reconciler.prepare_for_load(
                [optimizer],
                [cluster],
                [legacy_state],
                root_module=root_module,
            )

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(tuple(optimizer.param_groups), original_groups)
        for group, parameter_list, parameters in zip(
            optimizer.param_groups,
            original_parameter_lists,
            original_parameters,
            strict=True,
        ):
            self.assertIs(group["params"], parameter_list)
            self.assertEqual(tuple(group["params"]), parameters)
        self.assertIs(optimizer.state, original_state)
        self.assertEqual(tuple(optimizer.state.items()), original_state_items)
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rejects_external_aliases_across_optimizers(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        root_module.external = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([[10.0]])),
                nn.Parameter(torch.tensor([20.0])),
            ]
        )
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        initial_neuron.register_parameter(
            "tail_weight",
            nn.Parameter(torch.tensor([31.0])),
        )
        initial_neuron.register_parameter(
            "tail_bias",
            nn.Parameter(torch.tensor([32.0])),
        )
        dynamic_neuron.weight = root_module.external[0]
        dynamic_neuron.bias = root_module.external[1]
        dynamic_neuron.register_parameter(
            "tail_weight",
            nn.Parameter(torch.tensor([41.0])),
        )
        dynamic_neuron.register_parameter(
            "tail_bias",
            nn.Parameter(torch.tensor([42.0])),
        )
        first_external_parameter, second_external_parameter = root_module.external

        source_optimizers = [
            torch.optim.SGD(
                [
                    {
                        "params": [
                            initial_neuron.weight,
                            initial_neuron.tail_weight,
                            first_external_parameter,
                        ],
                        "lr": 0.1,
                    },
                    {"params": [dynamic_neuron.tail_weight], "lr": 0.1},
                ],
                momentum=0.9,
            ),
            torch.optim.SGD(
                [
                    {
                        "params": [
                            initial_neuron.bias,
                            initial_neuron.tail_bias,
                            second_external_parameter,
                        ],
                        "lr": 0.2,
                    },
                    {"params": [dynamic_neuron.tail_bias], "lr": 0.2},
                ],
                momentum=0.9,
            ),
        ]
        saved_states = [optimizer.state_dict() for optimizer in source_optimizers]
        optimizers = [
            torch.optim.SGD(
                [
                    initial_neuron.weight,
                    initial_neuron.tail_weight,
                    second_external_parameter,
                    dynamic_neuron.tail_weight,
                ],
                lr=0.1,
                momentum=0.9,
            ),
            torch.optim.SGD(
                [
                    initial_neuron.bias,
                    initial_neuron.tail_bias,
                    first_external_parameter,
                    dynamic_neuron.tail_bias,
                ],
                lr=0.2,
                momentum=0.9,
            ),
        ]
        original_group_lists = tuple(optimizer.param_groups for optimizer in optimizers)
        original_groups = tuple(
            tuple(optimizer.param_groups) for optimizer in optimizers
        )
        original_parameter_lists = tuple(
            tuple(group["params"] for group in optimizer.param_groups)
            for optimizer in optimizers
        )
        original_parameters = tuple(
            tuple(tuple(parameters) for parameters in parameter_lists)
            for parameter_lists in original_parameter_lists
        )
        original_states = tuple(optimizer.state for optimizer in optimizers)
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "multiple optimizers"):
            reconciler.prepare_for_load(
                optimizers,
                [cluster],
                saved_states,
                root_module=root_module,
            )

        for (
            optimizer,
            group_list,
            groups,
            parameter_lists,
            parameters_by_group,
            state,
        ) in zip(
            optimizers,
            original_group_lists,
            original_groups,
            original_parameter_lists,
            original_parameters,
            original_states,
            strict=True,
        ):
            self.assertIs(optimizer.param_groups, group_list)
            self.assertEqual(tuple(optimizer.param_groups), groups)
            for group, parameter_list, parameters in zip(
                optimizer.param_groups,
                parameter_lists,
                parameters_by_group,
                strict=True,
            ):
                self.assertIs(group["params"], parameter_list)
                self.assertEqual(tuple(group["params"]), parameters)
            self.assertIs(optimizer.state, state)
            self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_maps_external_alias_selected_for_suffix(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        root_module.other = nn.Linear(1, 1)
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        dynamic_neuron.weight = root_module.other.weight
        base_parameters = [initial_neuron.weight, initial_neuron.bias]
        dynamic_parameters = [dynamic_neuron.weight, dynamic_neuron.bias]
        source_optimizer = torch.optim.SGD(
            [
                {"params": base_parameters, "lr": 0.1},
                {"params": dynamic_parameters, "lr": 0.1},
            ],
            momentum=0.9,
        )
        expected_momentum_by_parameter_id = {}
        for index, parameter in enumerate(
            [*base_parameters, *dynamic_parameters],
            start=1,
        ):
            momentum = torch.full_like(parameter, float(index))
            source_optimizer.state[parameter]["momentum_buffer"] = momentum.clone()
            expected_momentum_by_parameter_id[id(parameter)] = momentum
        legacy_state = source_optimizer.state_dict()
        optimizer = torch.optim.SGD(
            cluster.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [optimizer],
            [cluster],
            [legacy_state],
            root_module=root_module,
        )
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in base_parameters],
                [id(parameter) for parameter in dynamic_parameters],
            ],
        )
        for parameter in (*base_parameters, *dynamic_parameters):
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                expected_momentum_by_parameter_id[id(parameter)],
            )

    def test_legacy_reconciler_preserves_named_groups_with_one_base_alias(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        root_module.other = nn.Linear(1, 1)
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        dynamic_neuron.weight = root_module.other.weight
        names_by_parameter_id = {
            id(parameter): name for name, parameter in root_module.named_parameters()
        }
        base_groups = [
            [initial_neuron.weight, dynamic_neuron.weight],
            [initial_neuron.bias, root_module.other.bias],
        ]
        dynamic_parameters = [dynamic_neuron.bias]

        def named_group(parameters, learning_rate):
            return {
                "params": parameters,
                "param_names": [
                    names_by_parameter_id[id(parameter)] for parameter in parameters
                ],
                "lr": learning_rate,
            }

        source_optimizer = torch.optim.SGD(
            [
                named_group(base_groups[0], 0.1),
                named_group(base_groups[1], 0.2),
                named_group(dynamic_parameters, 0.1),
            ],
            momentum=0.9,
        )
        expected_momentum_by_parameter_id = {}
        for index, parameter in enumerate(root_module.parameters(), start=1):
            momentum = torch.full_like(parameter, float(index))
            source_optimizer.state[parameter]["momentum_buffer"] = momentum.clone()
            expected_momentum_by_parameter_id[id(parameter)] = momentum
        legacy_state = source_optimizer.state_dict()
        optimizer = torch.optim.SGD(
            [
                named_group([*base_groups[0], *dynamic_parameters], 0.1),
                named_group(base_groups[1], 0.2),
            ],
            momentum=0.9,
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [optimizer],
            [cluster],
            [legacy_state],
            root_module=root_module,
        )
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        expected_groups = [*base_groups, dynamic_parameters]
        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in parameters]
                for parameters in expected_groups
            ],
        )
        self.assertEqual(
            [group["param_names"] for group in optimizer.param_groups],
            [
                [names_by_parameter_id[id(parameter)] for parameter in parameters]
                for parameters in expected_groups
            ],
        )
        for parameter in root_module.parameters():
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                expected_momentum_by_parameter_id[id(parameter)],
            )

    def test_legacy_reconciler_ignores_unoptimized_initial_alias_cluster(
        self,
    ) -> None:
        dynamic_cluster = _DynamicCluster()
        unoptimized_alias_cluster = _DynamicCluster()
        alias_initial_neuron = unoptimized_alias_cluster.cluster["neuron_0_0_0"]
        alias_dynamic_neuron = unoptimized_alias_cluster.cluster["neuron_1_0_0"]
        alias_dynamic_neuron.weight = alias_initial_neuron.weight
        alias_dynamic_neuron.bias = alias_initial_neuron.bias
        root_module = nn.Module()
        root_module.dynamic_cluster = dynamic_cluster
        root_module.unoptimized_alias_cluster = unoptimized_alias_cluster
        dynamic_initial_neuron = dynamic_cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = dynamic_cluster.cluster["neuron_1_0_0"]
        base_parameters = [
            dynamic_initial_neuron.weight,
            dynamic_initial_neuron.bias,
        ]
        dynamic_parameters = [dynamic_neuron.weight, dynamic_neuron.bias]
        source_optimizer = torch.optim.SGD(
            [
                {"params": base_parameters, "lr": 0.1},
                {"params": dynamic_parameters, "lr": 0.1},
            ],
            momentum=0.9,
        )
        expected_momentum_by_parameter_id = {}
        for index, parameter in enumerate(
            [*base_parameters, *dynamic_parameters],
            start=1,
        ):
            momentum = torch.full_like(parameter, float(index))
            source_optimizer.state[parameter]["momentum_buffer"] = momentum.clone()
            expected_momentum_by_parameter_id[id(parameter)] = momentum
        legacy_state = source_optimizer.state_dict()
        optimizer = torch.optim.SGD(
            dynamic_cluster.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [optimizer],
            [dynamic_cluster, unoptimized_alias_cluster],
            [legacy_state],
            root_module=root_module,
        )
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in base_parameters],
                [id(parameter) for parameter in dynamic_parameters],
            ],
        )
        for parameter in (*base_parameters, *dynamic_parameters):
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                expected_momentum_by_parameter_id[id(parameter)],
            )

    def test_legacy_reconciler_rejects_partial_external_alias_suffix(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.neuron_cluster = cluster
        root_module.external = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([[10.0]])),
                nn.Parameter(torch.tensor([20.0])),
            ]
        )
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        initial_neuron.register_parameter(
            "tail",
            nn.Parameter(torch.tensor([25.0])),
        )
        dynamic_neuron.weight = root_module.external[0]
        dynamic_neuron.bias = root_module.external[1]
        dynamic_neuron.register_parameter(
            "tail",
            nn.Parameter(torch.tensor([30.0])),
        )
        base_parameters = [
            initial_neuron.weight,
            initial_neuron.bias,
            initial_neuron.tail,
            dynamic_neuron.bias,
        ]
        dynamic_parameters = [dynamic_neuron.weight, dynamic_neuron.tail]
        source_optimizer = torch.optim.SGD(
            [
                {"params": base_parameters, "lr": 0.1},
                {"params": dynamic_parameters, "lr": 0.1},
            ],
            momentum=0.9,
        )
        legacy_state = source_optimizer.state_dict()
        optimizer = torch.optim.SGD(
            cluster.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        original_group_list = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        original_parameters = tuple(original_parameter_list)
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "base/suffix membership ambiguous"):
            reconciler.prepare_for_load(
                [optimizer],
                [cluster],
                [legacy_state],
                root_module=root_module,
            )

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(tuple(optimizer.param_groups), (original_group,))
        self.assertIs(optimizer.param_groups[0]["params"], original_parameter_list)
        self.assertEqual(tuple(original_parameter_list), original_parameters)
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_selects_one_suffix_owner_across_alias_clusters(
        self,
    ) -> None:
        dynamic_cluster = _DynamicCluster()
        alias_cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.dynamic_cluster = dynamic_cluster
        root_module.alias_cluster = alias_cluster
        root_module.external = nn.Linear(1, 1)
        alias_dynamic_neuron = alias_cluster.cluster["neuron_1_0_0"]
        alias_dynamic_neuron.weight = root_module.external.weight
        alias_dynamic_neuron.bias = root_module.external.bias
        dynamic_initial_neuron = dynamic_cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = dynamic_cluster.cluster["neuron_1_0_0"]
        alias_initial_neuron = alias_cluster.cluster["neuron_0_0_0"]
        base_parameters = [
            dynamic_initial_neuron.weight,
            dynamic_initial_neuron.bias,
            alias_initial_neuron.weight,
            alias_initial_neuron.bias,
            root_module.external.weight,
            root_module.external.bias,
        ]
        dynamic_parameters = [dynamic_neuron.weight, dynamic_neuron.bias]
        source_optimizer = torch.optim.SGD(
            [
                {"params": base_parameters, "lr": 0.1},
                {"params": dynamic_parameters, "lr": 0.1},
            ],
            momentum=0.9,
        )
        expected_momentum_by_parameter_id = {}
        for index, parameter in enumerate(root_module.parameters(), start=1):
            momentum = torch.full_like(parameter, float(index))
            source_optimizer.state[parameter]["momentum_buffer"] = momentum.clone()
            expected_momentum_by_parameter_id[id(parameter)] = momentum
        legacy_state = source_optimizer.state_dict()
        optimizer = torch.optim.SGD(
            root_module.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [optimizer],
            [dynamic_cluster, alias_cluster],
            [legacy_state],
            root_module=root_module,
        )
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(
            [
                [id(parameter) for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [id(parameter) for parameter in base_parameters],
                [id(parameter) for parameter in dynamic_parameters],
            ],
        )
        for parameter in root_module.parameters():
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                expected_momentum_by_parameter_id[id(parameter)],
            )

    def test_legacy_reconciler_rejects_competing_alias_clusters(
        self,
    ) -> None:
        first_cluster = _DynamicCluster()
        second_cluster = _DynamicCluster()
        root_module = nn.Module()
        root_module.first_cluster = first_cluster
        root_module.second_cluster = second_cluster
        root_module.external = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([[10.0]])),
                nn.Parameter(torch.tensor([[20.0]])),
                nn.Parameter(torch.tensor([30.0])),
            ]
        )
        first_external, second_external, third_external = root_module.external
        first_dynamic_neuron = first_cluster.cluster["neuron_1_0_0"]
        second_dynamic_neuron = second_cluster.cluster["neuron_1_0_0"]
        first_dynamic_neuron.weight = first_external
        first_dynamic_neuron.bias = first_external
        second_dynamic_neuron.weight = second_external
        second_dynamic_neuron.bias = third_external
        first_initial_neuron = first_cluster.cluster["neuron_0_0_0"]
        second_initial_neuron = second_cluster.cluster["neuron_0_0_0"]
        base_parameters = [
            first_initial_neuron.weight,
            first_initial_neuron.bias,
            second_initial_neuron.weight,
            second_initial_neuron.bias,
            first_external,
            third_external,
        ]
        source_optimizer = torch.optim.SGD(
            [
                {"params": base_parameters, "lr": 0.1},
                {"params": [second_external], "lr": 0.1},
            ],
            momentum=0.9,
        )
        for parameter, momentum_value in (
            (first_external, 11.0),
            (second_external, 22.0),
            (third_external, 33.0),
        ):
            source_optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                parameter,
                momentum_value,
            )
        legacy_state = source_optimizer.state_dict()
        optimizer = torch.optim.SGD(
            root_module.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        original_group_list = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        original_parameters = tuple(original_parameter_list)
        original_state = optimizer.state
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "alias membership across clusters"):
            reconciler.prepare_for_load(
                [optimizer],
                [first_cluster, second_cluster],
                [legacy_state],
                root_module=root_module,
            )

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(tuple(optimizer.param_groups), (original_group,))
        self.assertIs(optimizer.param_groups[0]["params"], original_parameter_list)
        self.assertEqual(tuple(original_parameter_list), original_parameters)
        self.assertIs(optimizer.state, original_state)
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_rejects_competing_fixed_initial_alias(
        self,
    ) -> None:
        first_cluster = _DynamicCluster()
        second_cluster = _DynamicCluster()
        unoptimized_cluster = _DynamicCluster(include_dynamic_neuron=False)
        root_module = nn.Module()
        root_module.first_cluster = first_cluster
        root_module.second_cluster = second_cluster
        root_module.unoptimized_cluster = unoptimized_cluster
        root_module.external = nn.ParameterList([nn.Parameter(torch.tensor([[10.0]]))])
        external_parameter = root_module.external[0]
        unoptimized_initial_parameter = unoptimized_cluster.cluster[
            "neuron_0_0_0"
        ].weight
        first_dynamic_neuron = first_cluster.cluster["neuron_1_0_0"]
        second_dynamic_neuron = second_cluster.cluster["neuron_1_0_0"]
        first_dynamic_neuron.weight = external_parameter
        first_dynamic_neuron.bias = external_parameter
        second_dynamic_neuron.weight = unoptimized_initial_parameter
        second_dynamic_neuron.bias = unoptimized_initial_parameter
        optimized_initial_parameters = [
            *first_cluster.cluster["neuron_0_0_0"].parameters(),
            *second_cluster.cluster["neuron_0_0_0"].parameters(),
        ]
        source_optimizer = torch.optim.SGD(
            [
                {
                    "params": [*optimized_initial_parameters, external_parameter],
                    "lr": 0.1,
                },
                {"params": [unoptimized_initial_parameter], "lr": 0.1},
            ],
            momentum=0.9,
        )
        for parameter, momentum_value in (
            (external_parameter, 11.0),
            (unoptimized_initial_parameter, 22.0),
        ):
            source_optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                parameter,
                momentum_value,
            )
        legacy_state = source_optimizer.state_dict()
        current_parameters = []
        current_parameter_ids: set[int] = set()
        for parameter in (*first_cluster.parameters(), *second_cluster.parameters()):
            if id(parameter) in current_parameter_ids:
                continue
            current_parameters.append(parameter)
            current_parameter_ids.add(id(parameter))
        optimizer = torch.optim.SGD(
            current_parameters,
            lr=0.1,
            momentum=0.9,
        )
        original_group_list = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        original_parameters = tuple(original_parameter_list)
        original_state = optimizer.state
        reconciler = NeuronOptimizerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "alias membership across clusters"):
            reconciler.prepare_for_load(
                [optimizer],
                [first_cluster, second_cluster, unoptimized_cluster],
                [legacy_state],
                root_module=root_module,
            )

        self.assertIs(optimizer.param_groups, original_group_list)
        self.assertEqual(tuple(optimizer.param_groups), (original_group,))
        self.assertIs(optimizer.param_groups[0]["params"], original_parameter_list)
        self.assertEqual(tuple(original_parameter_list), original_parameters)
        self.assertIs(optimizer.state, original_state)
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_legacy_reconciler_maps_tied_dynamic_roles_to_each_role_optimizer(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        dynamic_neuron.bias = dynamic_neuron.weight
        shared_dynamic_parameter = dynamic_neuron.weight

        def optimizer_and_legacy_state(
            initial_parameter,
            *,
            learning_rate: float,
            momentum_value: float,
        ):
            optimizer = torch.optim.SGD(
                [initial_parameter, shared_dynamic_parameter],
                lr=learning_rate,
                momentum=0.9,
            )
            optimizer.state[shared_dynamic_parameter]["momentum_buffer"] = (
                torch.full_like(shared_dynamic_parameter, momentum_value)
            )
            serialized_state = optimizer.state_dict()
            group_options = {
                name: value
                for name, value in serialized_state["param_groups"][0].items()
                if name != "params"
            }
            return optimizer, {
                "state": serialized_state["state"],
                "param_groups": [
                    {**group_options, "params": [0]},
                    {**group_options, "params": [1]},
                ],
            }

        weight_optimizer, weight_state = optimizer_and_legacy_state(
            initial_neuron.weight,
            learning_rate=0.1,
            momentum_value=3.0,
        )
        bias_optimizer, bias_state = optimizer_and_legacy_state(
            initial_neuron.bias,
            learning_rate=0.2,
            momentum_value=7.0,
        )
        cluster.requires_grad_(False)
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [weight_optimizer, bias_optimizer],
            [cluster],
            [weight_state, bias_state],
        )
        weight_optimizer.load_state_dict(weight_state)
        bias_optimizer.load_state_dict(bias_state)
        self.assertTrue(reconciler.complete_optimizer_load(weight_optimizer))
        self.assertTrue(reconciler.complete_optimizer_load(bias_optimizer))

        for optimizer, expected_momentum in (
            (weight_optimizer, 3.0),
            (bias_optimizer, 7.0),
        ):
            self.assertIs(
                optimizer.param_groups[1]["params"][0],
                shared_dynamic_parameter,
            )
            self.assertEqual(
                sum(
                    parameter is shared_dynamic_parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ),
                1,
            )
            torch.testing.assert_close(
                optimizer.state[shared_dynamic_parameter]["momentum_buffer"],
                torch.full_like(shared_dynamic_parameter, expected_momentum),
            )

    def test_legacy_reconciler_preserves_per_optimizer_tied_alias_order(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        dynamic_neuron = cluster.cluster["neuron_1_0_0"]
        initial_neuron.bias = nn.Parameter(torch.zeros(1, 1))
        dynamic_neuron.bias = nn.Parameter(torch.zeros(1, 1))
        initial_neuron.register_parameter("tail", nn.Parameter(torch.zeros(1, 1)))
        dynamic_neuron.register_parameter("tail", dynamic_neuron.weight)
        shared_dynamic_parameter = dynamic_neuron.weight
        weight_optimizer = torch.optim.SGD(
            [initial_neuron.weight, shared_dynamic_parameter],
            lr=0.1,
            momentum=0.9,
        )
        bias_optimizer = torch.optim.SGD(
            [
                initial_neuron.bias,
                initial_neuron.tail,
                dynamic_neuron.bias,
                shared_dynamic_parameter,
            ],
            lr=0.2,
            momentum=0.9,
        )
        for parameter, sentinel in (
            (dynamic_neuron.bias, 5.0),
            (shared_dynamic_parameter, 9.0),
        ):
            bias_optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                parameter,
                sentinel,
            )

        def legacy_state(optimizer, base_count: int):
            state = optimizer.state_dict()
            options = {
                name: value
                for name, value in state["param_groups"][0].items()
                if name != "params"
            }
            parameter_ids = state["param_groups"][0]["params"]
            return {
                "state": state["state"],
                "param_groups": [
                    {**options, "params": parameter_ids[:base_count]},
                    {**options, "params": parameter_ids[base_count:]},
                ],
            }

        weight_state = legacy_state(weight_optimizer, 1)
        bias_state = legacy_state(bias_optimizer, 2)
        cluster.requires_grad_(False)
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [weight_optimizer, bias_optimizer],
            [cluster],
            [weight_state, bias_state],
        )
        weight_optimizer.load_state_dict(weight_state)
        bias_optimizer.load_state_dict(bias_state)
        self.assertTrue(reconciler.complete_optimizer_load(weight_optimizer))
        self.assertTrue(reconciler.complete_optimizer_load(bias_optimizer))

        self.assertEqual(
            [id(parameter) for parameter in bias_optimizer.param_groups[1]["params"]],
            [id(dynamic_neuron.bias), id(shared_dynamic_parameter)],
        )
        torch.testing.assert_close(
            bias_optimizer.state[dynamic_neuron.bias]["momentum_buffer"],
            torch.full_like(dynamic_neuron.bias, 5.0),
        )
        torch.testing.assert_close(
            bias_optimizer.state[shared_dynamic_parameter]["momentum_buffer"],
            torch.full_like(shared_dynamic_parameter, 9.0),
        )

    def test_legacy_reconciler_completes_one_of_multiple_migrations(self) -> None:
        first_cluster, first_optimizer, first_state = self.legacy_fixture()
        second_cluster, second_optimizer, second_state = self.legacy_fixture()
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load(
            [first_optimizer, second_optimizer],
            [first_cluster, second_cluster],
            [first_state, second_state],
        )
        policy = reconciler.complete_optimizer_load(first_optimizer)

        self.assertEqual(policy, LegacyOptimizerAppendPolicy(1, 0, (0, 0)))
        self.assertFalse(reconciler.optimizer_requires_completion(first_optimizer))
        self.assertTrue(reconciler.optimizer_requires_completion(second_optimizer))
        reconciler.clear()
        self.assertEqual(len(second_optimizer.param_groups), 1)

    def test_legacy_reconciler_does_not_recreate_missing_base_groups(self) -> None:
        cluster, optimizer, saved_state = self.legacy_fixture()
        reconciler = NeuronOptimizerCheckpointReconciler()
        reconciler.prepare_for_load([optimizer], [cluster], [saved_state])

        optimizer.param_groups.clear()
        reconciler.clear()

        self.assertEqual(optimizer.param_groups, [])


class TestNeuronOptimizerNamedLayoutEdges(unittest.TestCase):
    @staticmethod
    def layout_fixture(*, split_groups: bool = False):
        module = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(index)))
                for index, name in enumerate(("a", "b", "c"), start=1)
            }
        )
        if split_groups:
            optimizer = torch.optim.SGD(
                [
                    {"params": [module["a"]], "lr": 0.1},
                    {"params": [module["b"], module["c"]], "lr": 0.2},
                ]
            )
        else:
            optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        saved_state = optimizer.state_dict()
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            [optimizer],
            [saved_state],
            {},
        )
        return module, optimizer, saved_state, layout

    def assert_prepare_rejected(
        self,
        module,
        optimizer,
        saved_state,
        layout,
        pattern: str,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, pattern):
            NeuronOptimizerNamedLayout().prepare_for_load(
                module,
                [optimizer],
                [saved_state],
                layout,
            )

    def test_completion_without_named_migration_returns_no_append_policy(
        self,
    ) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        layout_manager = NeuronOptimizerNamedLayout()

        self.assertFalse(layout_manager.optimizer_requires_completion(optimizer))
        self.assertIsNone(layout_manager.pending_append_policy(optimizer))
        self.assertIsNone(layout_manager.complete_optimizer_load(optimizer))
        self.assertFalse(layout_manager.optimizer_requires_completion(optimizer))

    def test_second_optimizer_apply_failure_rolls_back_every_saved_payload(
        self,
    ) -> None:
        module = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(index)))
                for index, name in enumerate(("a", "b", "c", "d"), start=1)
            }
        )
        source_optimizers = [
            torch.optim.SGD([module["a"], module["b"]], lr=0.1),
            torch.optim.SGD([module["c"], module["d"]], lr=0.2),
        ]
        saved_states = [optimizer.state_dict() for optimizer in source_optimizers]
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            source_optimizers,
            saved_states,
            {},
        )
        target_optimizers = [
            torch.optim.SGD([module["b"], module["a"]], lr=0.1),
            torch.optim.SGD([module["d"], module["c"]], lr=0.2),
        ]
        original_saved_ids = [
            tuple(state["param_groups"][0]["params"]) for state in saved_states
        ]
        saved_states[1]["param_groups"][0] = _ExplodingOnceDict(
            saved_states[1]["param_groups"][0]
        )
        layout_manager = NeuronOptimizerNamedLayout()

        with self.assertRaisesRegex(
            RuntimeError,
            "injected named-layout migration failure",
        ):
            layout_manager.prepare_for_load(
                module,
                target_optimizers,
                saved_states,
                layout,
            )

        self.assertEqual(
            [tuple(state["param_groups"][0]["params"]) for state in saved_states],
            original_saved_ids,
        )
        self.assertTrue(
            all(
                not layout_manager.optimizer_requires_completion(optimizer)
                for optimizer in target_optimizers
            )
        )

    def test_completing_one_named_optimizer_leaves_the_other_migration_pending(
        self,
    ) -> None:
        module = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(index)))
                for index, name in enumerate(("a", "b", "c", "d"), start=1)
            }
        )
        source_optimizers = [
            torch.optim.SGD([module["a"], module["b"]], lr=0.1, momentum=0.9),
            torch.optim.SGD([module["c"], module["d"]], lr=0.2, momentum=0.9),
        ]
        first_optimizer_parameter_ids = {id(module["a"]), id(module["b"])}
        for sentinel, parameter in enumerate(module.parameters(), start=11):
            owning_optimizer = (
                source_optimizers[0]
                if id(parameter) in first_optimizer_parameter_ids
                else source_optimizers[1]
            )
            owning_optimizer.state[parameter] = {
                "momentum_buffer": torch.full_like(parameter, float(sentinel))
            }
        saved_states = [optimizer.state_dict() for optimizer in source_optimizers]
        original_saved_ids = [
            tuple(state["param_groups"][0]["params"]) for state in saved_states
        ]
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            source_optimizers,
            saved_states,
            {},
        )
        target_optimizers = [
            torch.optim.SGD([module["b"], module["a"]], lr=0.1, momentum=0.9),
            torch.optim.SGD([module["d"], module["c"]], lr=0.2, momentum=0.9),
        ]
        layout_manager = NeuronOptimizerNamedLayout()
        layout_manager.prepare_for_load(
            module,
            target_optimizers,
            saved_states,
            layout,
        )

        target_optimizers[0].load_state_dict(saved_states[0])
        self.assertIsNone(layout_manager.complete_optimizer_load(target_optimizers[0]))

        self.assertFalse(
            layout_manager.optimizer_requires_completion(target_optimizers[0])
        )
        self.assertTrue(
            layout_manager.optimizer_requires_completion(target_optimizers[1])
        )
        self.assertEqual(
            tuple(saved_states[0]["param_groups"][0]["params"]),
            original_saved_ids[0],
        )
        self.assertEqual(saved_states[1]["param_groups"][0]["params"], [1, 0])
        torch.testing.assert_close(
            target_optimizers[0].state[module["a"]]["momentum_buffer"],
            torch.tensor(11.0),
        )
        torch.testing.assert_close(
            target_optimizers[0].state[module["b"]]["momentum_buffer"],
            torch.tensor(12.0),
        )

        layout_manager.clear()
        self.assertEqual(
            tuple(saved_states[1]["param_groups"][0]["params"]),
            original_saved_ids[1],
        )
        self.assertFalse(
            layout_manager.optimizer_requires_completion(target_optimizers[1])
        )

    def test_capture_rejects_misaligned_counts_names_and_membership(self) -> None:
        module, optimizer, saved_state, _ = self.layout_fixture()
        with self.assertRaisesRegex(RuntimeError, "optimizer counts differ"):
            NeuronOptimizerNamedLayout.capture(module, [optimizer], [], {})

        with self.assertRaisesRegex(RuntimeError, "parameter-group counts differ"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [{"state": {}, "param_groups": []}],
                {},
            )

        short_state = copy.deepcopy(saved_state)
        short_state["param_groups"][0]["params"].pop()
        with self.assertRaisesRegex(RuntimeError, "parameter-group size differs"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [short_state],
                {},
            )

        optimizer.param_groups[0]["param_names"] = []
        with self.assertRaisesRegex(RuntimeError, "live param_names"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [saved_state],
                {},
            )
        optimizer.param_groups[0].pop("param_names")

        named_state = copy.deepcopy(saved_state)
        named_state["param_groups"][0]["param_names"] = []
        with self.assertRaisesRegex(RuntimeError, "serialized param_names"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [named_state],
                {},
            )

        external_optimizer = torch.optim.SGD(
            [nn.Parameter(torch.tensor(4.0))],
            lr=0.1,
        )
        with self.assertRaisesRegex(RuntimeError, "registered on the Lightning module"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [external_optimizer],
                [external_optimizer.state_dict()],
                {},
            )

        duplicate_optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        duplicate_optimizer.param_groups[0]["params"].append(module["a"])
        with self.assertRaisesRegex(RuntimeError, "appears more than once"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [duplicate_optimizer],
                [duplicate_optimizer.state_dict()],
                {},
            )

    def test_prepare_rejects_malformed_layout_envelopes(self) -> None:
        module, optimizer, saved_state, layout = self.layout_fixture()
        with self.assertRaisesRegex(RuntimeError, "Invalid named"):
            NeuronOptimizerNamedLayout().prepare_for_load(
                module,
                [optimizer],
                [saved_state],
                None,
            )

        wrong_version = copy.deepcopy(layout)
        wrong_version["version"] = 2
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            wrong_version,
            "Unsupported named",
        )

        invalid_optimizers = copy.deepcopy(layout)
        invalid_optimizers["optimizers"] = [None]
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            invalid_optimizers,
            "Invalid named",
        )

        with self.assertRaisesRegex(RuntimeError, "optimizer counts differ"):
            NeuronOptimizerNamedLayout().prepare_for_load(
                module,
                [],
                [],
                layout,
            )

    def test_prepare_rejects_invalid_policy_groups_and_saved_names(self) -> None:
        module, optimizer, saved_state, layout = self.layout_fixture()

        invalid_policy = copy.deepcopy(layout)
        invalid_policy["optimizers"][0]["sync_policy"] = "unknown"
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            invalid_policy,
            "sync policy",
        )

        invalid_groups = copy.deepcopy(layout)
        invalid_groups["optimizers"][0]["parameter_names"] = []
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            invalid_groups,
            "group metadata",
        )

        invalid_append = copy.deepcopy(layout)
        invalid_append["optimizers"][0].update(
            {
                "sync_policy": "legacy_append",
                "legacy_base_group_count": False,
                "legacy_reference_group_index": 0,
            }
        )
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            invalid_append,
            "legacy append policy",
        )

        invalid_lineage = copy.deepcopy(layout)
        invalid_lineage["optimizers"][0].update(
            {
                "sync_policy": "legacy_append",
                "legacy_base_group_count": 1,
                "legacy_reference_group_index": 0,
                "legacy_group_reference_indices": [1],
            }
        )
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            invalid_lineage,
            "legacy group lineage",
        )

        invalid_name_shape = copy.deepcopy(layout)
        invalid_name_shape["optimizers"][0]["parameter_names"][0] = "a"
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            invalid_name_shape,
            "group metadata",
        )

        missing_name = copy.deepcopy(layout)
        missing_name["optimizers"][0]["parameter_names"][0][0] = "missing"
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            missing_name,
            "absent from the reconstructed model",
        )

        duplicate_name = copy.deepcopy(layout)
        duplicate_name["optimizers"][0]["parameter_names"][0][1] = "a"
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            duplicate_name,
            "duplicate parameter name",
        )

    def test_prepare_rejects_live_membership_and_group_drift(self) -> None:
        module, _, saved_state, layout = self.layout_fixture()
        external_optimizer = torch.optim.SGD(
            [nn.Parameter(torch.tensor(4.0))],
            lr=0.1,
        )
        self.assert_prepare_rejected(
            module,
            external_optimizer,
            saved_state,
            layout,
            "registered on the Lightning module",
        )

        missing_optimizer = torch.optim.SGD([module["a"], module["b"]], lr=0.1)
        self.assert_prepare_rejected(
            module,
            missing_optimizer,
            saved_state,
            layout,
            "membership differs",
        )

        split_optimizer = torch.optim.SGD(
            [
                {"params": [module["a"]], "lr": 0.1},
                {"params": [module["b"], module["c"]], "lr": 0.1},
            ]
        )
        self.assert_prepare_rejected(
            module,
            split_optimizer,
            saved_state,
            layout,
            "parameter-group counts differ",
        )

        split_module, _, split_state, split_layout = self.layout_fixture(
            split_groups=True
        )
        regrouped_optimizer = torch.optim.SGD(
            [
                {
                    "params": [split_module["a"], split_module["b"]],
                    "lr": 0.1,
                },
                {"params": [split_module["c"]], "lr": 0.2},
            ]
        )
        self.assert_prepare_rejected(
            split_module,
            regrouped_optimizer,
            split_state,
            split_layout,
            "parameter-group membership differs",
        )


class TestNeuronOptimizerSchedulerEdges(unittest.TestCase):
    @staticmethod
    def nested_scheduler_fixture():
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(2)]
        optimizer = torch.optim.SGD([parameters[0]], lr=0.1)
        first = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        second = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([first, second])
        saved_state = scheduler.state_dict()
        reference_options = {
            name: value
            for name, value in optimizer.param_groups[0].items()
            if name != "params"
        }
        optimizer.add_param_group({**reference_options, "params": [parameters[1]]})
        return optimizer, scheduler, first, second, saved_state

    def test_nested_scheduler_reconciliation_and_removal_are_recursive(self) -> None:
        optimizer, scheduler, first, second, saved_state = (
            self.nested_scheduler_fixture()
        )
        reconciler = NeuronSchedulerCheckpointReconciler()
        binding = SchedulerGroupLoadBinding(
            scheduler=scheduler,
            saved_state=saved_state,
            optimizer=optimizer,
            policy=LegacyOptimizerAppendPolicy(1, 0),
            target_group_count=2,
        )
        foreign_optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(3.0))], lr=0.1)

        self.assertFalse(reconciler.optimizer_requires_completion(foreign_optimizer))
        reconciler.prepare_for_load([binding])
        reconciler.mark_optimizer_loaded(foreign_optimizer)
        self.assertEqual(first.base_lrs, [0.1, 0.1])
        self.assertEqual(second.base_lrs, [0.1, 0.1])
        reconciler.clear()
        self.assertEqual(first.base_lrs, [0.1])
        self.assertEqual(second.base_lrs, [0.1])

        reconcile_scheduler_group_count(
            scheduler,
            None,
            LegacyOptimizerAppendPolicy(1, 0),
            target_group_count=2,
        )
        remove_scheduler_groups(
            scheduler,
            (1,),
            previous_group_count=2,
        )
        self.assertEqual(first.base_lrs, [0.1])
        self.assertEqual(second.base_lrs, [0.1])

    def test_nested_scheduler_rejects_saved_child_count_drift(self) -> None:
        optimizer, scheduler, _, _, saved_state = self.nested_scheduler_fixture()
        saved_state["_schedulers"].pop()
        binding = SchedulerGroupLoadBinding(
            scheduler=scheduler,
            saved_state=saved_state,
            optimizer=optimizer,
            policy=LegacyOptimizerAppendPolicy(1, 0),
            target_group_count=2,
        )

        with self.assertRaisesRegex(RuntimeError, "child counts differ"):
            NeuronSchedulerCheckpointReconciler().prepare_for_load([binding])

    def test_scheduler_snapshot_handles_aliased_child_payloads(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(2)]
        optimizer = torch.optim.SGD([parameters[0]], lr=0.1)
        first = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        second = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([first, second])
        saved_state = scheduler.state_dict()
        saved_state["_schedulers"][1] = saved_state["_schedulers"][0]
        optimizer.add_param_group({"params": [parameters[1]], "lr": 0.1})
        binding = SchedulerGroupLoadBinding(
            scheduler=scheduler,
            saved_state=saved_state,
            optimizer=optimizer,
            policy=LegacyOptimizerAppendPolicy(1, 0),
            target_group_count=2,
        )
        reconciler = NeuronSchedulerCheckpointReconciler()

        reconciler.prepare_for_load([binding])
        reconciler.clear()

        self.assertIs(saved_state["_schedulers"][0], saved_state["_schedulers"][1])
        self.assertEqual(saved_state["_schedulers"][0]["base_lrs"], [0.1])

    def test_scheduler_reconciliation_rolls_back_apply_failures(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        original_base_lrs = _ExplodingList(scheduler.base_lrs)
        scheduler.base_lrs = original_base_lrs
        binding = SchedulerGroupLoadBinding(
            scheduler=scheduler,
            saved_state=None,
            optimizer=optimizer,
            policy=LegacyOptimizerAppendPolicy(1, 0),
            target_group_count=2,
        )
        reconciler = NeuronSchedulerCheckpointReconciler()

        with self.assertRaisesRegex(RuntimeError, "injected scheduler mutation"):
            reconciler.prepare_for_load([binding])

        self.assertIs(scheduler.base_lrs, original_base_lrs)
        self.assertEqual(scheduler.base_lrs, [0.1])
        self.assertFalse(reconciler.optimizer_requires_completion(optimizer))

    def test_scheduler_partial_commit_is_partitioned_by_optimizer_identity(
        self,
    ) -> None:
        first_parameters = [nn.Parameter(torch.tensor(value)) for value in (1.0, 2.0)]
        first_optimizer = torch.optim.SGD([first_parameters[0]], lr=0.1)
        first_scheduler = torch.optim.lr_scheduler.StepLR(
            first_optimizer,
            step_size=1,
        )
        first_saved_state = copy.deepcopy(first_scheduler.state_dict())
        first_live_base_lrs = first_scheduler.base_lrs
        first_live_last_lrs = first_scheduler._last_lr
        first_payload_base_lrs = first_saved_state["base_lrs"]
        first_payload_last_lrs = first_saved_state["_last_lr"]

        second_parameters = [nn.Parameter(torch.tensor(value)) for value in (3.0, 4.0)]
        second_optimizer = torch.optim.SGD([second_parameters[0]], lr=0.3)
        second_scheduler = torch.optim.lr_scheduler.StepLR(
            second_optimizer,
            step_size=1,
        )
        second_saved_state = copy.deepcopy(second_scheduler.state_dict())
        second_live_base_lrs = second_scheduler.base_lrs
        second_live_last_lrs = second_scheduler._last_lr
        second_payload_base_lrs = second_saved_state["base_lrs"]
        second_payload_last_lrs = second_saved_state["_last_lr"]
        second_original_last_epoch = second_scheduler.last_epoch
        second_payload_original_last_epoch = second_saved_state["last_epoch"]

        first_optimizer.add_param_group(
            {"params": [first_parameters[1]], "lr": 0.2},
        )
        second_optimizer.add_param_group(
            {"params": [second_parameters[1]], "lr": 0.4},
        )
        reconciler = NeuronSchedulerCheckpointReconciler()
        reconciler.prepare_for_load(
            [
                SchedulerGroupLoadBinding(
                    scheduler=first_scheduler,
                    saved_state=first_saved_state,
                    optimizer=first_optimizer,
                    policy=LegacyOptimizerAppendPolicy(1, 0),
                    target_group_count=2,
                ),
                SchedulerGroupLoadBinding(
                    scheduler=second_scheduler,
                    saved_state=second_saved_state,
                    optimizer=second_optimizer,
                    policy=LegacyOptimizerAppendPolicy(1, 0),
                    target_group_count=2,
                ),
            ],
        )

        self.assertTrue(reconciler.optimizer_requires_completion(first_optimizer))
        self.assertTrue(reconciler.optimizer_requires_completion(second_optimizer))
        self.assertIs(first_scheduler.base_lrs, first_live_base_lrs)
        self.assertIs(first_scheduler._last_lr, first_live_last_lrs)
        self.assertIs(first_saved_state["base_lrs"], first_payload_base_lrs)
        self.assertIs(first_saved_state["_last_lr"], first_payload_last_lrs)
        self.assertEqual(first_scheduler.base_lrs, [0.1, 0.1])
        self.assertEqual(first_scheduler._last_lr, [0.1, 0.1])
        self.assertEqual(first_saved_state["base_lrs"], [0.1, 0.1])
        self.assertEqual(first_saved_state["_last_lr"], [0.1, 0.1])
        self.assertEqual(second_scheduler.base_lrs, [0.3, 0.3])
        self.assertEqual(second_scheduler._last_lr, [0.3, 0.3])
        self.assertEqual(second_saved_state["base_lrs"], [0.3, 0.3])
        self.assertEqual(second_saved_state["_last_lr"], [0.3, 0.3])

        first_scheduler.last_epoch = 101
        first_saved_state["last_epoch"] = 102
        second_scheduler.last_epoch = 201
        second_saved_state["last_epoch"] = 202
        reconciler.mark_optimizer_loaded(first_optimizer)
        reconciler.commit_loaded()

        self.assertFalse(reconciler.optimizer_requires_completion(first_optimizer))
        self.assertTrue(reconciler.optimizer_requires_completion(second_optimizer))
        reconciler.clear()

        self.assertIs(first_scheduler.base_lrs, first_live_base_lrs)
        self.assertIs(first_scheduler._last_lr, first_live_last_lrs)
        self.assertIs(first_saved_state["base_lrs"], first_payload_base_lrs)
        self.assertIs(first_saved_state["_last_lr"], first_payload_last_lrs)
        self.assertEqual(first_scheduler.base_lrs, [0.1, 0.1])
        self.assertEqual(first_scheduler._last_lr, [0.1, 0.1])
        self.assertEqual(first_saved_state["base_lrs"], [0.1, 0.1])
        self.assertEqual(first_saved_state["_last_lr"], [0.1, 0.1])
        self.assertEqual(first_scheduler.last_epoch, 101)
        self.assertEqual(first_saved_state["last_epoch"], 102)
        self.assertIs(second_scheduler.base_lrs, second_live_base_lrs)
        self.assertIs(second_scheduler._last_lr, second_live_last_lrs)
        self.assertIs(second_saved_state["base_lrs"], second_payload_base_lrs)
        self.assertIs(second_saved_state["_last_lr"], second_payload_last_lrs)
        self.assertEqual(second_scheduler.base_lrs, [0.3])
        self.assertEqual(second_scheduler._last_lr, [0.3])
        self.assertEqual(second_saved_state["base_lrs"], [0.3])
        self.assertEqual(second_saved_state["_last_lr"], [0.3])
        self.assertEqual(second_scheduler.last_epoch, second_original_last_epoch)
        self.assertEqual(
            second_saved_state["last_epoch"],
            second_payload_original_last_epoch,
        )
        self.assertFalse(reconciler.optimizer_requires_completion(second_optimizer))

    def test_scheduler_group_count_handles_extension_and_mixed_removal(
        self,
    ) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(2)]
        optimizer = torch.optim.SGD([parameters[0]], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        preflight_scheduler_group_extension(
            scheduler,
            previous_group_count=1,
            reference_group_index=0,
        )
        reference_options = {
            name: value
            for name, value in optimizer.param_groups[0].items()
            if name != "params"
        }
        optimizer.add_param_group(
            {**reference_options, "params": [parameters[1]]},
        )
        extend_scheduler_for_new_group(
            scheduler,
            previous_group_count=1,
            reference_group_index=0,
        )

        self.assertEqual(scheduler.base_lrs, [0.1, 0.1])
        self.assertEqual(scheduler.get_last_lr(), [0.1, 0.1])

        scheduler.base_lrs.pop()
        preflight_scheduler_group_removal(
            scheduler,
            (1,),
            previous_group_count=2,
        )
        optimizer.param_groups.pop()
        remove_scheduler_groups(
            scheduler,
            (1,),
            previous_group_count=2,
        )

        self.assertEqual(scheduler.base_lrs, [0.1])
        self.assertEqual(scheduler.get_last_lr(), [0.1])

    def test_nonadjacent_group_removal_preserves_cyclic_scheduler_alignment(
        self,
    ) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(4)]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [parameter],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                }
                for parameter, learning_rate, momentum, weight_decay in zip(
                    parameters,
                    (0.10, 0.20, 0.30, 0.40),
                    (0.71, 0.72, 0.73, 0.74),
                    (0.001, 0.002, 0.003, 0.004),
                    strict=True,
                )
            ]
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=[0.01, 0.02, 0.03, 0.04],
            max_lr=[0.11, 0.12, 0.13, 0.14],
            step_size_up=2,
            cycle_momentum=True,
            base_momentum=[0.61, 0.62, 0.63, 0.64],
            max_momentum=[0.91, 0.92, 0.93, 0.94],
        )
        removed_group_indices = (1, 3)
        surviving_group_indices = (0, 2)
        original_groups = tuple(optimizer.param_groups)
        expected_groups = tuple(
            original_groups[index] for index in surviving_group_indices
        )
        expected_group_options = tuple(
            {
                name: value
                for name, value in original_groups[index].items()
                if name != "params"
            }
            for index in surviving_group_indices
        )
        scheduler_sequences = {
            sequence_name: getattr(scheduler, sequence_name)
            for sequence_name in (
                "base_lrs",
                "_last_lr",
                "max_lrs",
                "base_momentums",
                "max_momentums",
            )
        }
        expected_scheduler_values = {
            sequence_name: tuple(values[index] for index in surviving_group_indices)
            for sequence_name, values in scheduler_sequences.items()
        }

        preflight_scheduler_group_removal(
            scheduler,
            removed_group_indices,
            previous_group_count=4,
        )
        optimizer.param_groups[:] = expected_groups
        remove_scheduler_groups(
            scheduler,
            removed_group_indices,
            previous_group_count=4,
        )

        self.assertEqual(tuple(optimizer.param_groups), expected_groups)
        for group, expected_options in zip(
            optimizer.param_groups,
            expected_group_options,
            strict=True,
        ):
            self.assertEqual(
                {name: value for name, value in group.items() if name != "params"},
                expected_options,
            )
        for sequence_name, original_list in scheduler_sequences.items():
            self.assertIs(getattr(scheduler, sequence_name), original_list)
            self.assertEqual(
                tuple(getattr(scheduler, sequence_name)),
                expected_scheduler_values[sequence_name],
            )

        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()

        expected_learning_rates = (0.06, 0.08)
        expected_momentums = (0.76, 0.78)
        for group, expected_learning_rate, expected_momentum in zip(
            optimizer.param_groups,
            expected_learning_rates,
            expected_momentums,
            strict=True,
        ):
            self.assertAlmostEqual(group["lr"], expected_learning_rate, places=12)
            self.assertAlmostEqual(group["momentum"], expected_momentum, places=12)
        for actual, expected in zip(
            scheduler.get_last_lr(),
            expected_learning_rates,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=12)

        post_step_scheduler_sequences = {
            sequence_name: getattr(scheduler, sequence_name)
            for sequence_name in scheduler_sequences
        }
        post_step_scheduler_values = {
            name: tuple(values)
            for name, values in post_step_scheduler_sequences.items()
        }
        preflight_scheduler_group_removal(
            scheduler,
            removed_group_indices,
            previous_group_count=4,
        )
        remove_scheduler_groups(
            scheduler,
            removed_group_indices,
            previous_group_count=4,
        )

        self.assertEqual(tuple(optimizer.param_groups), expected_groups)
        for sequence_name, post_step_list in post_step_scheduler_sequences.items():
            self.assertIs(getattr(scheduler, sequence_name), post_step_list)
            self.assertEqual(
                tuple(getattr(scheduler, sequence_name)),
                post_step_scheduler_values[sequence_name],
            )

    def test_reduce_on_plateau_min_lrs_follow_group_extension_and_removal(
        self,
    ) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(3)]
        optimizer = torch.optim.SGD([parameters[0]], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=0,
            min_lr=0.01,
        )
        original_min_lrs = scheduler.min_lrs
        original_last_lrs = scheduler._last_lr

        for parameter, learning_rate, reference_group_index in (
            (parameters[1], 0.2, 0),
            (parameters[2], 0.3, 1),
        ):
            previous_group_count = len(optimizer.param_groups)
            preflight_scheduler_group_extension(
                scheduler,
                previous_group_count=previous_group_count,
                reference_group_index=reference_group_index,
            )
            reference_options = {
                name: value
                for name, value in optimizer.param_groups[reference_group_index].items()
                if name != "params"
            }
            optimizer.add_param_group(
                {
                    **reference_options,
                    "params": [parameter],
                    "lr": learning_rate,
                }
            )
            extend_scheduler_for_new_group(
                scheduler,
                previous_group_count=previous_group_count,
                reference_group_index=reference_group_index,
            )

        self.assertIs(scheduler.min_lrs, original_min_lrs)
        self.assertIs(scheduler._last_lr, original_last_lrs)
        self.assertEqual(scheduler.min_lrs, [0.01, 0.01, 0.01])
        self.assertEqual(scheduler._last_lr, [0.1, 0.1, 0.1])

        preflight_scheduler_group_removal(
            scheduler,
            (1,),
            previous_group_count=3,
        )
        optimizer.param_groups.pop(1)
        remove_scheduler_groups(
            scheduler,
            (1,),
            previous_group_count=3,
        )

        self.assertIs(scheduler.min_lrs, original_min_lrs)
        self.assertIs(scheduler._last_lr, original_last_lrs)
        self.assertEqual(scheduler.min_lrs, [0.01, 0.01])
        self.assertEqual(scheduler._last_lr, [0.1, 0.1])
        scheduler.step(1.0)
        scheduler.step(2.0)
        self.assertEqual(len(scheduler.get_last_lr()), 2)

    def test_scheduler_validation_rejects_invalid_policy_lengths_and_removal(
        self,
    ) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        with self.assertRaisesRegex(RuntimeError, "reconciliation policy"):
            reconcile_scheduler_group_count(
                scheduler,
                None,
                LegacyOptimizerAppendPolicy(2, 0),
                target_group_count=1,
            )

        scheduler.base_lrs.extend([0.2, 0.3])
        with self.assertRaisesRegex(RuntimeError, "base_lrs has 3 entries"):
            preflight_scheduler_group_extension(
                scheduler,
                previous_group_count=1,
                reference_group_index=0,
            )
        scheduler.base_lrs[:] = [0.1]

        with self.assertRaisesRegex(RuntimeError, "group-removal indices"):
            preflight_scheduler_group_removal(
                scheduler,
                (0, 0),
                previous_group_count=1,
            )


class TestNeuronOptimizerSyncCheckpointEdges(unittest.TestCase):
    @staticmethod
    def trainer(optimizers):
        return SimpleNamespace(
            optimizers=optimizers,
            state=SimpleNamespace(fn=TrainerFn.FITTING),
            lr_scheduler_configs=[],
        )

    def test_callback_ignores_non_optimizer_checkpoint_payloads(self) -> None:
        module = nn.Linear(1, 1)
        trainer = self.trainer([])
        callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = {"optimizer_states": {}}

        callback.on_load_checkpoint(trainer, module, checkpoint)
        callback.on_save_checkpoint(trainer, module, checkpoint)

        self.assertNotIn(OPTIMIZER_LAYOUT_CHECKPOINT_KEY, checkpoint)

    def test_callback_preserves_native_checkpoint_for_external_parameter(self) -> None:
        module = nn.Linear(1, 1)
        external_parameter = nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam(
            [*module.parameters(), external_parameter],
            lr=0.03,
        )
        loss = module(torch.ones(1, 1)).sum() + external_parameter.sum()
        loss.backward()
        optimizer.step()
        saved_optimizer_states = [copy.deepcopy(optimizer.state_dict())]
        checkpoint = {
            "optimizer_states": saved_optimizer_states,
            OPTIMIZER_LAYOUT_CHECKPOINT_KEY: {"stale": True},
        }

        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_save_checkpoint(
            self.trainer([optimizer]),
            module,
            checkpoint,
        )

        self.assertNotIn(OPTIMIZER_LAYOUT_CHECKPOINT_KEY, checkpoint)
        self.assertIs(checkpoint["optimizer_states"], saved_optimizer_states)

        restored_module = nn.Linear(1, 1)
        restored_external_parameter = nn.Parameter(torch.tensor([-1.0]))
        restored_optimizer = torch.optim.Adam(
            [*restored_module.parameters(), restored_external_parameter],
            lr=0.5,
        )
        restored_trainer = self.trainer([restored_optimizer])
        restored_callback = NeuronClusterOptimizerSyncCallback()
        restored_callback.on_load_checkpoint(
            restored_trainer,
            restored_module,
            checkpoint,
        )
        restored_optimizer.load_state_dict(saved_optimizer_states[0])
        restored_callback.on_fit_start(restored_trainer, restored_module)

        source_parameters = [*module.parameters(), external_parameter]
        restored_parameters = [
            *restored_module.parameters(),
            restored_external_parameter,
        ]
        for source_parameter, restored_parameter in zip(
            source_parameters,
            restored_parameters,
            strict=True,
        ):
            with self.subTest(parameter_shape=tuple(source_parameter.shape)):
                source_state = optimizer.state[source_parameter]
                restored_state = restored_optimizer.state[restored_parameter]
                self.assertEqual(source_state.keys(), restored_state.keys())
                for state_name in source_state:
                    torch.testing.assert_close(
                        restored_state[state_name],
                        source_state[state_name],
                    )

    def test_callback_falls_back_for_native_only_optimizer_layouts(self) -> None:
        duplicate_module = nn.Linear(1, 1, bias=False)
        with self.assertWarnsRegex(UserWarning, "duplicate parameters"):
            duplicate_optimizer = torch.optim.SGD(
                [duplicate_module.weight, duplicate_module.weight],
                lr=0.1,
            )
        misnamed_module = nn.Linear(1, 1)
        misnamed_optimizer = torch.optim.SGD(
            [
                {
                    "params": list(misnamed_module.parameters()),
                    "param_names": ["custom_name"],
                }
            ],
            lr=0.2,
        )

        for case_name, module, optimizer in (
            ("duplicate-parameter", duplicate_module, duplicate_optimizer),
            ("misaligned-param-names", misnamed_module, misnamed_optimizer),
        ):
            with self.subTest(case_name=case_name):
                saved_optimizer_states = [optimizer.state_dict()]
                checkpoint = {
                    "optimizer_states": saved_optimizer_states,
                    OPTIMIZER_LAYOUT_CHECKPOINT_KEY: {"stale": True},
                }

                NeuronClusterOptimizerSyncCallback().on_save_checkpoint(
                    self.trainer([optimizer]),
                    module,
                    checkpoint,
                )

                self.assertNotIn(OPTIMIZER_LAYOUT_CHECKPOINT_KEY, checkpoint)
                self.assertIs(
                    checkpoint["optimizer_states"],
                    saved_optimizer_states,
                )

    def test_reused_callback_does_not_transfer_legacy_policy_by_reused_id(
        self,
    ) -> None:
        def make_optimizer() -> tuple[nn.Parameter, torch.optim.Optimizer]:
            parameter = nn.Parameter(torch.tensor(1.0))
            return parameter, torch.optim.SGD([parameter], lr=0.1)

        _, old_optimizer = make_optimizer()
        new_parameter, new_optimizer = make_optimizer()
        callback = NeuronClusterOptimizerSyncCallback()
        reused_optimizer_id = id(new_optimizer)
        callback._legacy_append_policies[reused_optimizer_id] = (
            LegacyOptimizerAppendPolicy(1, 0)
        )
        callback._legacy_append_policy_owners[reused_optimizer_id] = weakref.ref(
            old_optimizer
        )
        callback.on_fit_end(SimpleNamespace(), nn.Linear(1, 1))
        new_module = nn.Module()
        new_module.register_parameter("value", new_parameter)
        trainer = self.trainer([new_optimizer])
        callback.on_fit_start(trainer, new_module)
        checkpoint = {"optimizer_states": [new_optimizer.state_dict()]}
        callback.on_save_checkpoint(trainer, new_module, checkpoint)

        self.assertEqual(callback._legacy_append_policies, {})
        self.assertEqual(
            checkpoint[OPTIMIZER_LAYOUT_CHECKPOINT_KEY]["optimizers"][0]["sync_policy"],
            "role",
        )

    def test_pickled_callback_rekeys_policy_to_same_optimizer_object(self) -> None:
        module = nn.Module()
        parameter = nn.Parameter(torch.tensor(1.0))
        module.register_parameter("value", parameter)
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        original_optimizer_id = id(optimizer)
        callback = NeuronClusterOptimizerSyncCallback()
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )

        restored_callback, restored_module, restored_optimizer = pickle.loads(
            pickle.dumps((callback, module, optimizer))
        )
        self.assertNotEqual(id(restored_optimizer), original_optimizer_id)
        trainer = self.trainer([restored_optimizer])
        restored_callback.on_fit_start(trainer, restored_module)
        checkpoint = {"optimizer_states": [restored_optimizer.state_dict()]}
        restored_callback.on_save_checkpoint(
            trainer,
            restored_module,
            checkpoint,
        )

        restored_optimizer_id = id(restored_optimizer)
        self.assertIs(
            restored_callback._legacy_append_policy_owners[restored_optimizer_id](),
            restored_optimizer,
        )
        self.assertEqual(
            checkpoint[OPTIMIZER_LAYOUT_CHECKPOINT_KEY]["optimizers"][0]["sync_policy"],
            "legacy_append",
        )

    def test_pickled_preload_policy_rolls_back_with_optimizer_identity(self) -> None:
        module = nn.Module()
        parameter = nn.Parameter(torch.tensor(1.0))
        module.register_parameter("value", parameter)
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        callback = NeuronClusterOptimizerSyncCallback()
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )
        callback._pre_load_legacy_append_policies = dict(
            callback._legacy_append_policies
        )
        callback._pre_load_legacy_append_policy_owners = dict(
            callback._legacy_append_policy_owners
        )
        callback._legacy_append_policies.clear()
        callback._legacy_append_policy_owners.clear()

        restored_callback, restored_module, restored_optimizer = pickle.loads(
            pickle.dumps((callback, module, optimizer))
        )
        restored_callback.on_exception(
            SimpleNamespace(),
            restored_module,
            RuntimeError("cancel load"),
        )
        checkpoint = {"optimizer_states": [restored_optimizer.state_dict()]}
        restored_callback.on_save_checkpoint(
            self.trainer([restored_optimizer]),
            restored_module,
            checkpoint,
        )

        self.assertEqual(
            checkpoint[OPTIMIZER_LAYOUT_CHECKPOINT_KEY]["optimizers"][0]["sync_policy"],
            "legacy_append",
        )

    def test_fit_end_does_not_retain_legacy_optimizer_or_parameters(self) -> None:
        module = nn.Module()
        parameter = nn.Parameter(torch.tensor(1.0))
        module.register_parameter("value", parameter)
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        parameter_reference = weakref.ref(parameter)
        optimizer_reference = weakref.ref(optimizer)
        callback = NeuronClusterOptimizerSyncCallback()
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )

        callback.on_fit_end(SimpleNamespace(), module)
        del module, optimizer, parameter
        gc.collect()

        self.assertIsNone(optimizer_reference())
        self.assertIsNone(parameter_reference())
        self.assertFalse(
            any(
                owner_reference() is not None
                for owner_reference in (callback._legacy_append_policy_owners.values())
            )
        )

    def test_callback_rolls_back_invalid_named_layout(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        trainer = self.trainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        original_group = optimizer.param_groups[0]
        checkpoint = {
            "optimizer_states": [optimizer.state_dict()],
            OPTIMIZER_LAYOUT_CHECKPOINT_KEY: {
                "version": 999,
                "optimizers": [],
            },
        }

        with self.assertRaisesRegex(RuntimeError, "Unsupported named"):
            callback.on_load_checkpoint(trainer, module, checkpoint)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIs(optimizer.param_groups[0], original_group)

    def test_callback_rejects_scheduler_count_drift(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainer = self.trainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = {
            "optimizer_states": [optimizer.state_dict()],
            "lr_schedulers": [],
        }

        with self.assertRaisesRegex(RuntimeError, "scheduler counts differ"):
            callback.on_load_checkpoint(trainer, module, checkpoint)

    def test_callback_ignores_foreign_scheduler_and_duplicate_hook_request(
        self,
    ) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        foreign_parameter = nn.Parameter(torch.tensor(2.0))
        foreign_optimizer = torch.optim.SGD([foreign_parameter], lr=0.2)
        foreign_scheduler = torch.optim.lr_scheduler.StepLR(
            foreign_optimizer,
            step_size=1,
        )
        trainer = self.trainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=foreign_scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = {
            "optimizer_states": [optimizer.state_dict()],
            "lr_schedulers": [foreign_scheduler.state_dict()],
        }

        callback.on_load_checkpoint(trainer, module, checkpoint)
        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        callback.on_train_start(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 1)


if __name__ == "__main__":
    unittest.main()
