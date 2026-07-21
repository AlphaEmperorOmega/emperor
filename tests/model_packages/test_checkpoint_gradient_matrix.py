from __future__ import annotations

import hashlib
import io
import json
import unittest

import torch

from models.catalog import discover_model_packages, model_package

CHECKPOINT_REPRESENTATIVES = {
    "bert": "bert/linear",
    "experts": "experts/linear",
    "gpt": "gpt/linear",
    "linears": "linears/linear",
    "neuron": "neuron/linear",
    "parametric": "parametric/parametric_vector",
    "transformer": "transformer/linear",
    "vit": "vit/linear",
}
EXPECTED_STATE_TOPOLOGY_DIGESTS = {
    "bert/linear": "2364fb8a377892d1c8f69f73dfa917624e6ceb4ba3ae3018f708c5070b88fe77",
    "experts/linear": (
        "a1fabf86f7668f95cbeecce54693e51562d69a28eb307bca350bd46233ee05c4"
    ),
    "gpt/linear": "54c78200bb9174a43c25b60945796f9575af9dceb22a9ae8a6a9fd55a64205de",
    "linears/linear": (
        "b2581d7521ca8f3662e48bd8bfa81414d694d734633d5a901e5b3133d9163f56"
    ),
    "neuron/linear": "13dbc77f380c5cb91816620c01167d6904e9f50084d43d6b291b6ff587d0d5a4",
    "parametric/parametric_vector": (
        "813b5c878296fb417680b6e2d08124704da450c12a144911f0fa006e7ae13902"
    ),
    "transformer/linear": (
        "ad9f6e30e19b24247cba778fb75e3ee25c17eecd56551e595c00e39034e39db7"
    ),
    "vit/linear": "c1394b219688fffe506dd30dbc2d4c4ceafcc4fdbef293f95e19860684b3d357",
}
PERSISTENT_BUFFER_FAMILIES = {"experts", "neuron", "parametric"}
GRADIENT_PACKAGES = (
    "experts/linear_adaptive",
    "neuron/linear",
    "neuron/linear_adaptive",
    "neuron/expert_linear",
    "neuron/expert_linear_adaptive",
    "vit/linear_adaptive",
)
NEURON_GRADIENT_PACKAGES = (
    "neuron/linear",
    "neuron/linear_adaptive",
    "neuron/expert_linear",
    "neuron/expert_linear_adaptive",
)


class ModelPackageCheckpointGradientMatrixTests(unittest.TestCase):
    def test_representative_package_from_every_family_round_trips_checkpoint(
        self,
    ) -> None:
        self.assertEqual(
            set(CHECKPOINT_REPRESENTATIVES),
            {package.model_type for package in discover_model_packages()},
        )

        for family, model_id in CHECKPOINT_REPRESENTATIVES.items():
            with self.subTest(family=family, model_package=model_id):
                package = model_package(model_id)
                self.assertIsNotNone(package)
                config = package.build_configurations()[0]
                model = package.build_model(config)
                state = model.state_dict()
                topology = [
                    (name, str(tensor.dtype), list(tensor.shape))
                    for name, tensor in sorted(state.items())
                ]
                topology_digest = hashlib.sha256(
                    json.dumps(
                        topology,
                        separators=(",", ":"),
                        sort_keys=False,
                    ).encode("utf-8")
                ).hexdigest()
                state_names = set(state)
                parameter_names = {
                    name for name, _ in model.named_parameters(remove_duplicate=False)
                }
                buffer_names = {
                    name for name, _ in model.named_buffers(remove_duplicate=False)
                }

                self.assertTrue(parameter_names)
                self.assertTrue(state_names)
                self.assertEqual(
                    topology_digest,
                    EXPECTED_STATE_TOPOLOGY_DIGESTS[model_id],
                )
                self.assertTrue(parameter_names.issubset(state_names))
                self.assertTrue(state_names.issubset(parameter_names | buffer_names))
                if family in PERSISTENT_BUFFER_FAMILIES:
                    self.assertTrue(state_names & buffer_names)

                checkpoint = io.BytesIO()
                torch.save(state, checkpoint)
                checkpoint.seek(0)
                restored_state = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                restored_model = package.build_model(config)
                incompatible = restored_model.load_state_dict(
                    restored_state,
                    strict=True,
                )

                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                restored = restored_model.state_dict()
                self.assertEqual(set(restored), state_names)
                for name, tensor in state.items():
                    with self.subTest(state=name):
                        self.assertEqual(restored[name].dtype, tensor.dtype)
                        self.assertEqual(restored[name].shape, tensor.shape)
                        torch.testing.assert_close(restored[name], tensor)

    def test_required_packages_produce_finite_end_to_end_gradients(self) -> None:
        for model_id in GRADIENT_PACKAGES:
            with self.subTest(model_package=model_id):
                torch.manual_seed(23)
                package = model_package(model_id)
                self.assertIsNotNone(package)
                dataset = package.dataset_metadata[package.default_experiment_task][0]
                config = package.build_configurations(dataset=dataset)[0]
                model = package.build_model(config)
                inputs = torch.randn(
                    2,
                    dataset.num_channels,
                    dataset.default_height,
                    dataset.default_width,
                )

                output = model(inputs)
                if isinstance(output, tuple):
                    logits, auxiliary_loss = output
                else:
                    logits = output
                    auxiliary_loss = torch.zeros((), device=logits.device)
                loss = logits.float().square().mean() + auxiliary_loss.float()

                self.assertTrue(loss.requires_grad)
                self.assertTrue(torch.isfinite(loss.detach()).item())
                loss.backward()

                gradients = {
                    name: parameter.grad
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                self.assertTrue(gradients)
                for name, gradient in gradients.items():
                    with self.subTest(parameter=name):
                        self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(
                    any(
                        gradient.abs().sum().item() > 0.0
                        for gradient in gradients.values()
                    )
                )
                self._assert_nonzero_boundary_gradient(
                    gradients,
                    prefixes=("input_model.", "patch."),
                    role="input",
                )
                self._assert_nonzero_boundary_gradient(
                    gradients,
                    prefixes=("output_model.", "output."),
                    role="output",
                )
                if model_id in NEURON_GRADIENT_PACKAGES:
                    self._assert_all_role_parameters_receive_material_gradients(
                        model,
                        prefixes=("neuron_cluster.halting_model.",),
                        role="StickBreaking ponder gate",
                    )

    def test_neuron_packages_propagate_task_gradients_through_called_parts(
        self,
    ) -> None:
        for model_id in NEURON_GRADIENT_PACKAGES:
            with self.subTest(model_package=model_id):
                torch.manual_seed(23)
                package = model_package(model_id)
                self.assertIsNotNone(package)
                dataset = package.dataset_metadata[package.default_experiment_task][0]
                config = package.build_configurations(dataset=dataset)[0]
                model = package.build_model(config).double().eval()
                cluster = model.neuron_cluster
                called_nuclei: set[str] = set()
                called_terminals: set[str] = set()
                hook_handles = []
                for neuron_name, neuron in cluster.cluster.items():
                    hook_handles.append(
                        neuron.nucleus.register_forward_hook(
                            self._record_module_call(called_nuclei, neuron_name)
                        )
                    )
                    hook_handles.append(
                        neuron.terminal.register_forward_hook(
                            self._record_module_call(called_terminals, neuron_name)
                        )
                    )

                total_input_values = (
                    2
                    * dataset.num_channels
                    * dataset.default_height
                    * dataset.default_width
                )
                inputs = torch.linspace(
                    -0.75,
                    1.25,
                    total_input_values,
                    dtype=torch.float64,
                ).reshape(
                    2,
                    dataset.num_channels,
                    dataset.default_height,
                    dataset.default_width,
                )
                inputs.requires_grad_()
                try:
                    logits, _auxiliary_loss = model(inputs)
                finally:
                    for handle in hook_handles:
                        handle.remove()

                task_loss = logits.square().mean()
                task_loss.backward()

                self._assert_material_gradient(inputs, role="model input")
                self.assertEqual(inputs.grad.shape, inputs.shape)
                self.assertEqual(inputs.grad.dtype, inputs.dtype)
                self.assertEqual(inputs.grad.device, inputs.device)
                self._assert_all_role_parameters_receive_material_gradients(
                    model,
                    prefixes=("input_model.",),
                    role="input boundary",
                )
                self._assert_all_role_parameters_receive_material_gradients(
                    model,
                    prefixes=("output_model.",),
                    role="output boundary",
                )
                self._assert_all_role_parameters_receive_material_gradients(
                    model,
                    prefixes=("neuron_cluster.entry_sampler.",),
                    role="entry router",
                )
                self._assert_all_role_parameters_are_connected_and_finite(
                    model,
                    prefixes=("neuron_cluster.halting_model.",),
                    role="StickBreaking task path",
                )

                self.assertTrue(called_nuclei)
                self.assertTrue(called_terminals)
                self.assertTrue(called_terminals.issubset(called_nuclei))
                for neuron_name, neuron in cluster.cluster.items():
                    nucleus_parameters = tuple(neuron.nucleus.parameters())
                    terminal_parameters = tuple(neuron.terminal.parameters())
                    self.assertTrue(nucleus_parameters)
                    self.assertTrue(terminal_parameters)
                    if neuron_name in called_nuclei:
                        self._assert_called_module_has_material_gradient(
                            nucleus_parameters,
                            role=f"{neuron_name} nucleus",
                        )
                    else:
                        self._assert_parameters_are_inactive(
                            nucleus_parameters,
                            role=f"{neuron_name} nucleus",
                        )
                    if neuron_name in called_terminals:
                        for parameter in terminal_parameters:
                            self._assert_material_gradient(
                                parameter,
                                role=f"{neuron_name} terminal router",
                            )
                    else:
                        self._assert_parameters_are_inactive(
                            terminal_parameters,
                            role=f"{neuron_name} terminal router",
                        )

    @staticmethod
    def _record_module_call(called_modules: set[str], module_name: str):
        def record_call(_module, _inputs, _output) -> None:
            called_modules.add(module_name)

        return record_call

    def _assert_all_role_parameters_receive_material_gradients(
        self,
        model,
        *,
        prefixes: tuple[str, ...],
        role: str,
    ) -> None:
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(prefixes)
        ]
        self.assertTrue(parameters, f"no {role} parameters")
        for parameter in parameters:
            self._assert_material_gradient(parameter, role=role)

    def _assert_called_module_has_material_gradient(
        self,
        parameters: tuple[torch.nn.Parameter, ...],
        *,
        role: str,
    ) -> None:
        connected_gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        self.assertTrue(connected_gradients, f"{role} was called but disconnected")
        for gradient in connected_gradients:
            self.assertTrue(torch.isfinite(gradient).all().item(), role)
        self.assertTrue(
            any(
                self._is_material_gradient(gradient) for gradient in connected_gradients
            ),
            f"{role} was called but every connected gradient was zero",
        )

    def _assert_all_role_parameters_are_connected_and_finite(
        self,
        model,
        *,
        prefixes: tuple[str, ...],
        role: str,
    ) -> None:
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(prefixes)
        ]
        self.assertTrue(parameters, f"no {role} parameters")
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad, f"{role} is disconnected")
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), role)

    def _assert_parameters_are_inactive(
        self,
        parameters: tuple[torch.nn.Parameter, ...],
        *,
        role: str,
    ) -> None:
        self.assertTrue(
            all(parameter.grad is None for parameter in parameters),
            f"{role} was not called but received a task gradient",
        )

    def _assert_material_gradient(self, tensor, *, role: str) -> None:
        gradient = tensor.grad
        self.assertIsNotNone(gradient, f"{role} is disconnected")
        self.assertTrue(torch.isfinite(gradient).all().item(), role)
        self.assertTrue(
            self._is_material_gradient(gradient),
            f"{role} gradient is numerically zero",
        )

    @staticmethod
    def _is_material_gradient(gradient: torch.Tensor) -> bool:
        threshold = 32 * torch.finfo(gradient.dtype).eps
        return bool(gradient.detach().abs().max().item() > threshold)

    def _assert_nonzero_boundary_gradient(
        self,
        gradients: dict[str, torch.Tensor],
        *,
        prefixes: tuple[str, ...],
        role: str,
    ) -> None:
        role_gradients = [
            gradient
            for name, gradient in gradients.items()
            if name.startswith(prefixes)
        ]
        self.assertTrue(role_gradients, f"no {role} boundary gradient")
        self.assertTrue(
            any(gradient.abs().sum().item() > 0.0 for gradient in role_gradients),
            f"all {role} boundary gradients were zero",
        )


if __name__ == "__main__":
    unittest.main()
