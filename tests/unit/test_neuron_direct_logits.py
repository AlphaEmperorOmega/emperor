import copy
from unittest.mock import patch

import torch

from unit.test_neuron import NeuronTestCase


class TestNeuronDirectLogits(NeuronTestCase):
    def direct_terminal_config(self, noisy=True, width=None):
        count = self.terminal_total_connections()
        sampler = self.sampler_config(num_experts=count, top_k=2, router_config=None)
        sampler.noisy_topk_flag = noisy
        return self.terminal_config(
            input_dim=width if width is not None else count * (2 if noisy else 1),
            sampler_config=sampler,
        )

    def test_noisy_routerless_terminal_accepts_double_width(self):
        config = self.direct_terminal_config()
        terminal = config.build().eval()
        source = torch.randn(2, config.input_dim, requires_grad=True)
        _, probabilities, coordinates, _ = terminal(source)
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertEqual(coordinates.shape, (2, 2, 3))
        probabilities.sum().backward()
        self.assertTrue(torch.isfinite(source.grad).all())

    def test_direct_logit_noise_equation_and_gradients_in_train_and_eval(self):
        for noisy in (False, True):
            for training in (False, True):
                with self.subTest(noisy=noisy, training=training):
                    config = self.direct_terminal_config(noisy=noisy)
                    terminal = config.build().train(training)
                    source = (
                        torch.linspace(-1, 1, 2 * config.input_dim)
                        .reshape(2, -1)
                        .requires_grad_()
                    )
                    reference = source.detach().clone().requires_grad_()
                    count = config.sampler_config.num_experts
                    self.assertEqual(
                        config.sampler_config.required_logit_width(), config.input_dim
                    )
                    torch.manual_seed(17)
                    expected_logits = reference[:, :count]
                    if noisy and training:
                        expected_logits = expected_logits + (
                            reference[:, count:].sigmoid() + 0.01
                        ) * torch.randn_like(expected_logits)
                    expected = expected_logits.softmax(-1).topk(2, dim=-1)
                    torch.manual_seed(17)
                    _, probabilities, coordinates, _ = terminal(source)
                    torch.testing.assert_close(probabilities, expected.values)
                    torch.testing.assert_close(
                        coordinates, terminal.neuron_connections[expected.indices]
                    )
                    probabilities.sum().backward()
                    expected.values.sum().backward()
                    torch.testing.assert_close(source.grad, reference.grad)
                    if noisy:
                        self.assertEqual(
                            bool(source.grad[:, count:].abs().sum() > 0), training
                        )

    def test_invalid_direct_width_rejects_before_sampler_construction_or_rng(self):
        for noisy, width in ((False, 26), (False, 54), (True, 27), (True, 53)):
            with self.subTest(noisy=noisy, width=width):
                config = self.direct_terminal_config(noisy=noisy, width=width)
                original = copy.deepcopy(config)
                random_state = torch.get_rng_state().clone()
                with patch(
                    "emperor.sampler._sampler.SamplerModel.__init__",
                    side_effect=AssertionError("sampler constructed"),
                ):
                    with self.assertRaisesRegex(ValueError, "required logit width"):
                        config.build()
                self.assertEqual(config, original)
                torch.testing.assert_close(torch.get_rng_state(), random_state)

    def test_sampler_runtime_uses_configured_logit_width(self):
        for noisy in (False, True):
            config = self.direct_terminal_config(noisy=noisy).sampler_config
            sampler = config.build()
            with self.assertRaisesRegex(
                ValueError, f"expected {config.required_logit_width()}"
            ):
                sampler.sample_probabilities_and_indices(
                    torch.zeros(2, config.required_logit_width() - 1)
                )
