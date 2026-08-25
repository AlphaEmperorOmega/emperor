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
    "mlp_mixer": "mlp_mixer/linear",
    "neuron": "neuron/linear",
    "parametric": "parametric/parametric_vector",
    "transformer": "transformer/linear",
    "vit": "vit/linear",
}
EXPECTED_STATE_TOPOLOGY_DIGESTS = {
    "bert/expert_linear": (
        "0e34f906ff9c15ab2b692422ec3f920a75e2099ea0221db6033ce647558963de"
    ),
    "bert/expert_linear_adaptive": (
        "0e34f906ff9c15ab2b692422ec3f920a75e2099ea0221db6033ce647558963de"
    ),
    "bert/linear": "4da5c3efec5f6bf0647684b176eb79d5393af751c7dba55d712295c6aae7a578",
    "bert/linear_adaptive": (
        "f49635e8c2dec1594c993bcb6ecfdbc0e5fb49871fdc0800a356727fe5fcb3ac"
    ),
    "experts/linear": (
        "ef1cca635a4c85f84b6dcdb7d051dc987a9f9e67d480bed848173b2c4be3dfc2"
    ),
    "experts/linear_adaptive": (
        "9c128ad37e9dab9f2ef405a93ad424f171a287369efea2b02ff5f58dc0fb8404"
    ),
    "gpt/expert_linear": (
        "a2179fc50c406447af062b21d0533e3aa8bcd9045c9690c7fc934c7f3d1aaed3"
    ),
    "gpt/expert_linear_adaptive": (
        "a2179fc50c406447af062b21d0533e3aa8bcd9045c9690c7fc934c7f3d1aaed3"
    ),
    "gpt/linear": "94311c337b236db9e5a4b3ff22487340d2a8dbf67aed7ad12f264a5eafbcd385",
    "gpt/linear_adaptive": (
        "a7e10c274e8bcc628a90c19f19057c4acf104ca02109bc34f2e525331561dbf2"
    ),
    "linears/linear": (
        "efbffe89fe687a962fd9fd1b1c13f089a4739265333fd0e96545155280d7b807"
    ),
    "linears/linear_adaptive": (
        "efbffe89fe687a962fd9fd1b1c13f089a4739265333fd0e96545155280d7b807"
    ),
    "mlp_mixer/expert_linear": (
        "f3110f1d187bbe6186ce5096b44a1a80d4ad1309e1d51e96d22e843971c7320d"
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "5f52c2b87c72e873ba0b73b0d556e7677424415e126334b25a2937d8a7284920"
    ),
    "mlp_mixer/linear": (
        "0afa286265806894707abe3ed75352f402718c9b0fab6f9966a8ec1855678f9d"
    ),
    "mlp_mixer/linear_adaptive": (
        "56e3ef8de17c47284dec93bac4817e28c55a185f1887c969abd7af558eee0bf4"
    ),
    "neuron/expert_linear": (
        "1f2ffe2f7f3d551009917c36c3c218b25728a09058fe4ac6148a6b88f2c124e5"
    ),
    "neuron/expert_linear_adaptive": (
        "08cdd31dd48a29cfcd799efa802ee04e4d9d6ba747f5a8d85f92a0ba28788b26"
    ),
    "neuron/linear": "0d19dac15b982b01a20bda8b3b73f6e7e7b9291257e723a42fafb5aece2c2cfe",
    "neuron/linear_adaptive": (
        "0d19dac15b982b01a20bda8b3b73f6e7e7b9291257e723a42fafb5aece2c2cfe"
    ),
    "parametric/parametric_generator": (
        "86ee37a7fe511e6bcdb99a6d95604c6a27c590f040bf7a6dd6765e8be0f930ca"
    ),
    "parametric/parametric_matrix": (
        "fcc0060a3a33635fa427ff438ea1d4838d360fa9827791cb2e8977a88ff203f5"
    ),
    "parametric/parametric_vector": (
        "813b5c878296fb417680b6e2d08124704da450c12a144911f0fa006e7ae13902"
    ),
    "transformer/expert_linear": (
        "206455f265c52feb87899de5ae49475a783c17d46387db4f34cebc6844d02b2e"
    ),
    "transformer/expert_linear_adaptive": (
        "206455f265c52feb87899de5ae49475a783c17d46387db4f34cebc6844d02b2e"
    ),
    "transformer/linear": (
        "fde98a20e29b45427d4d2c63cfbd028593817cbf5f49fdfde06ebd629ceb8189"
    ),
    "transformer/linear_adaptive": (
        "fde98a20e29b45427d4d2c63cfbd028593817cbf5f49fdfde06ebd629ceb8189"
    ),
    "vit/expert_linear": (
        "2629c55dec5ce67fe7aaae8c28098d431ce7eea7d14ddad8cd6eb0e687e5b997"
    ),
    "vit/expert_linear_adaptive": (
        "2629c55dec5ce67fe7aaae8c28098d431ce7eea7d14ddad8cd6eb0e687e5b997"
    ),
    "vit/linear": "75d24a36a83addef48dff10016d13d3a2b4f07e15f1dacd12ebf10662bc917b9",
    "vit/linear_adaptive": (
        "31800b04e35bdf5ce67f93d0b77fa0cf57c030481e22bcc9f86c4d9a0b7308e2"
    ),
}
PERSISTENT_BUFFER_FAMILIES = {"experts", "neuron", "parametric"}
GRADIENT_PACKAGES = (
    "experts/linear_adaptive",
    "neuron/linear",
    "neuron/linear_adaptive",
    "neuron/expert_linear",
    "neuron/expert_linear_adaptive",
    "vit/linear_adaptive",
    "mlp_mixer/linear",
    "mlp_mixer/linear_adaptive",
    "mlp_mixer/expert_linear",
    "mlp_mixer/expert_linear_adaptive",
)
NEURON_GRADIENT_PACKAGES = (
    "neuron/linear",
    "neuron/linear_adaptive",
    "neuron/expert_linear",
    "neuron/expert_linear_adaptive",
)


class ModelPackageCheckpointGradientMatrixTests(unittest.TestCase):
    def test_every_package_round_trips_a_strict_cpu_checkpoint(self) -> None:
        discovered_packages = discover_model_packages()
        self.assertEqual(
            set(CHECKPOINT_REPRESENTATIVES),
            {package.identity.model_type for package in discovered_packages},
        )
        self.assertEqual(
            set(EXPECTED_STATE_TOPOLOGY_DIGESTS),
            {package.catalog_key for package in discovered_packages},
        )

        for package in discovered_packages:
            family = package.identity.model_type
            model_id = package.catalog_key
            with self.subTest(family=family, model_package=model_id):
                config = package.build_configuration()
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
                for tensor in state.values():
                    self.assertEqual(tensor.device.type, "cpu")

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
                        self.assertEqual(restored[name].device.type, "cpu")
                        torch.testing.assert_close(restored[name], tensor)

    def test_required_packages_produce_finite_end_to_end_gradients(self) -> None:
        for model_id in GRADIENT_PACKAGES:
            with self.subTest(model_package=model_id):
                torch.manual_seed(23)
                package = model_package(model_id)
                self.assertIsNotNone(package)
                dataset = package.dataset_metadata[package.default_experiment_task][0]
                config = package.build_configuration(dataset=dataset)
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
                config = package.build_configuration(dataset=dataset)
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
