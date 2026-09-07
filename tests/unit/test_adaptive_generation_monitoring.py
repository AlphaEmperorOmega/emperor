"""Base parameters, generated outputs, and selected routes stay observable."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterMonitorCallback,
    WeightBankUtilizationMonitorCallback,
)
from emperor.linears import LinearMonitorCallback
from support.adaptive_generation import bias_mixture_config, weight_mixture_config


class ObservedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=weight_mixture_config(top_k=1),
                bias_config=bias_mixture_config(top_k=1),
            ),
        ).build()
        self.global_step = 0
        self.logger = None
        self.metrics = {}

    def log(self, name, value, **kwargs):
        assert value.grad_fn is None and not value.requires_grad
        self.metrics[name] = value


@pytest.mark.parametrize("exception", [False, True])
def test_paired_parameter_and_actual_selected_bank_observations_cleanup(exception):
    module = ObservedModule()
    trainer = SimpleNamespace(global_step=0, training=True)
    callbacks = [
        AdaptiveParameterMonitorCallback(1),
        WeightBankUtilizationMonitorCallback(1, log_per_slot_scalars=True),
    ]
    augmentation = module.linear.adaptive_behaviour
    matrix_models = (augmentation.weight_model, augmentation.bias_model)
    original_instance_keys = [set(vars(model)) for model in matrix_models]
    with torch.no_grad():
        router = augmentation.weight_model.sampler.router.model[0].model
        router.weight_params.zero_()
        router.bias_params.copy_(torch.tensor([0.1, 0.2, 0.7]).log())
        bias_router = augmentation.bias_model.sampler.router.model[0].model
        bias_router.weight_params.zero_()
        bias_router.bias_params.copy_(torch.tensor([0.8, 0.15, 0.05]).log())
    for callback in callbacks:
        callback.on_fit_start(trainer, module)
        callback.on_fit_start(
            trainer, module
        )  # A repeated setup must not duplicate hooks.
    module.linear(torch.randn(4, 2, requires_grad=True))
    callbacks[1].on_train_batch_end(trainer, module, None, None, 0)
    prefix = "linear.adaptive_behaviour/"
    assert prefix + "weight/batch/output_mean" in module.metrics
    assert prefix + "bias/batch/output_mean" in module.metrics
    assert prefix + "weight/batch/weight_bank_l2_norm" in module.metrics
    bank_prefix = "linear.adaptive_behaviour.weight_model/bank/"
    assert module.metrics[bank_prefix + "active_slots"] == 1
    assert module.metrics[bank_prefix + "slot_0/utilization"] == 0
    torch.testing.assert_close(
        module.metrics[bank_prefix + "slot_2/utilization"], torch.tensor(0.7)
    )
    bias_bank_prefix = "linear.adaptive_behaviour.bias_model/bank/"
    assert module.metrics[bias_bank_prefix + "active_slots"] == 1
    assert module.metrics[bias_bank_prefix + "slot_2/utilization"] == 0
    torch.testing.assert_close(
        module.metrics[bias_bank_prefix + "slot_0/utilization"], torch.tensor(0.8)
    )
    assert not callbacks[1]._last_bank_logits
    for callback in callbacks:
        if exception:
            callback.on_exception(trainer, module, RuntimeError("interrupted"))
        else:
            callback.on_fit_end(trainer, module)
        assert not callback._hooks
    assert not callbacks[1]._utilization_history
    assert not callbacks[1]._method_restorers
    for model, original_keys in zip(matrix_models, original_instance_keys, strict=True):
        assert set(vars(model)) == original_keys
    assert all(
        not child._forward_hooks and not child._forward_pre_hooks
        for child in module.modules()
    )
    module.linear(torch.randn(4, 2))
    assert not callbacks[1]._last_bank_logits


@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_bank_monitor_observes_sampler_routes_without_changing_outputs_or_gradients(
    config_factory, top_k, monkeypatch
):
    bank = config_factory(top_k=top_k).build()
    module = nn.Module()
    module.bank = bank
    context = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], requires_grad=True)
    expected = bank(torch.zeros_like(bank.parameter_bank[0]), context)
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), (bank.parameter_bank, context), allow_unused=True
    )
    callback = WeightBankUtilizationMonitorCallback(1)
    samples = []
    original_sample = bank.sampler.sample_probabilities_and_indices

    def observe_sample(input_matrix):
        assert input_matrix is context
        sample = original_sample(input_matrix)
        samples.append(sample)
        return sample

    monkeypatch.setattr(
        bank.sampler, "sample_probabilities_and_indices", observe_sample
    )
    callback.on_fit_start(None, module)
    try:
        output = bank(torch.zeros_like(bank.parameter_bank[0]), context)
        gradients = torch.autograd.grad(
            output.square().sum(), (bank.parameter_bank, context), allow_unused=True
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        for actual, expected_gradient in zip(
            gradients, expected_gradients, strict=True
        ):
            if expected_gradient is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected_gradient, rtol=0, atol=0)
        assert len(samples) == 1
        probabilities, indices, _, _ = samples[0]
        expected_distribution = torch.zeros(2, 3)
        for row in range(2):
            for selection in range(top_k):
                expert_index = (
                    selection
                    if indices is None
                    else indices.reshape(2, top_k)[row, selection]
                )
                expected_distribution[row, expert_index] += probabilities.reshape(
                    2, top_k
                )[row, selection].detach()
        captured = callback._last_bank_logits["bank"]
        assert captured.grad_fn is None and not captured.requires_grad
        torch.testing.assert_close(captured, expected_distribution)
    finally:
        callback.on_fit_end(None, module)


def test_linear_monitor_keeps_bank_layer_activations_and_base_diagnostics():
    module = ObservedModule()
    trainer = SimpleNamespace(global_step=0, training=True)
    callback = LinearMonitorCallback(log_every_n_steps=1)
    callback.on_fit_start(trainer, module)
    callback.on_train_batch_start(trainer, module, None, 0)
    module.linear(torch.randn(4, 2, requires_grad=True)).sum().backward()
    callback.on_before_optimizer_step(trainer, module, None)
    # Exercise callback lifecycle only: no optimizer is constructed or stepped.
    trainer.global_step = module.global_step = 1
    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert any(
        name.startswith("linear/") and "input" in name for name in module.metrics
    )
    assert any(
        name.startswith("linear/") and "output" in name for name in module.metrics
    )
    assert any(
        name.startswith("linear/") and "weight" in name for name in module.metrics
    )
    callback.on_exception(trainer, module, RuntimeError("interrupted"))
    assert all(not child._forward_hooks for child in module.modules())
