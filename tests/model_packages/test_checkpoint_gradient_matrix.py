from __future__ import annotations

import io
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
PERSISTENT_BUFFER_FAMILIES = {"experts", "neuron", "parametric"}
GRADIENT_PACKAGES = (
    "experts/linear_adaptive",
    "neuron/linear",
    "neuron/linear_adaptive",
    "neuron/expert_linear",
    "neuron/expert_linear_adaptive",
    "vit/linear_adaptive",
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
                state_names = set(state)
                parameter_names = {
                    name
                    for name, _ in model.named_parameters(remove_duplicate=False)
                }
                buffer_names = {
                    name for name, _ in model.named_buffers(remove_duplicate=False)
                }

                self.assertTrue(parameter_names)
                self.assertTrue(state_names)
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
                dataset = package.dataset_metadata[
                    package.default_experiment_task
                ][0]
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
