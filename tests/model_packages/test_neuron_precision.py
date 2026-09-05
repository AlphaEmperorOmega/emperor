"""Small real Model Package forward/backward precision contracts; no updates."""

import importlib

import pytest
import torch

from emperor.neuron import TerminalRoutingTreeDepthOptions
from emperor.sampler import SamplerModel
from model_packages.test_neuron_terminal_routing_tree_runtime import PACKAGE_BUILDERS


def _build_model(package, builder_name, beam_width, tree_depth, **overrides):
    torch.manual_seed(17)
    values = {
        "input_dim": 4,
        "hidden_dim": 4,
        "output_dim": 2,
        "cluster_x_axis_total_neurons": 1,
        "cluster_y_axis_total_neurons": 1,
        "cluster_z_axis_total_neurons": 1,
        "cluster_initial_x_axis_total_neurons": 1,
        "cluster_initial_y_axis_total_neurons": 1,
        "cluster_initial_z_axis_total_neurons": 1,
        "cluster_max_steps": 2,
        "cluster_beam_width": beam_width,
        "cluster_growth_threshold": None,
        "cluster_pruning_threshold": None,
        "cluster_terminal_sampler_num_topk_samples": 0,
        "cluster_terminal_sampler_noisy_topk_flag": False,
    }
    values.update(overrides)
    if tree_depth is not None:
        values.update(
            {
                "cluster_terminal_routing_tree_depth": TerminalRoutingTreeDepthOptions(
                    tree_depth
                ),
                "cluster_terminal_routing_tree_level_1_branch_count": 2,
                "cluster_terminal_routing_tree_level_1_top_k": 1,
            }
        )
        if tree_depth == 3:
            values.update(
                {
                    "cluster_terminal_routing_tree_level_2_branch_count": 2,
                    "cluster_terminal_routing_tree_level_2_top_k": 1,
                }
            )
    runtime = importlib.import_module(
        f"models.neuron.{package}.runtime_defaults"
    ).runtime_from_flat(values)
    builder = getattr(
        importlib.import_module(f"models.neuron.{package}.config_builder"), builder_name
    )
    return (
        importlib.import_module(f"models.neuron.{package}.model")
        .Model(builder(runtime=runtime).build())
        .eval()
    )


@pytest.mark.parametrize("package,builder_name", PACKAGE_BUILDERS)
@pytest.mark.parametrize("beam_width", [1, 2])
@pytest.mark.parametrize("tree_depth", [None, 2, 3])
@pytest.mark.parametrize("precision", [torch.bfloat16, torch.float64])
def test_neuron_packages_support_cpu_autocast(
    package, builder_name, beam_width, tree_depth, precision
):
    model = _build_model(package, builder_name, beam_width, tree_depth)
    logit_margins = []
    # Sparse expert choices are discontinuous at ties. Use a real, trainable
    # router with separated logits for the cross-precision gradient oracle.
    with torch.no_grad():
        for name, sampler in model.named_modules():
            if not isinstance(sampler, SamplerModel) or ".nucleus." not in name:
                continue
            final = sampler.router.model.layers[-1].model
            final.weight_params.mul_(0.02)
            final.bias_params.copy_(
                torch.linspace(-1, 1, sampler.num_experts).to(final.bias_params)
            )

            def observe_margin(
                module, inputs, output, top_k=sampler.sampler_config.top_k
            ):
                logits = (
                    output.hidden.detach().float().sort(dim=-1, descending=True).values
                )
                if 0 < top_k < logits.shape[-1] and logits.numel():
                    logit_margins.append(
                        (logits[..., top_k - 1] - logits[..., top_k]).min().item()
                    )

            sampler.router.model.register_forward_hook(observe_margin)
    source = torch.tensor(
        [[0.1, 0.3, -0.2, 0.4], [-0.1, 0.2, 0.3, -0.4]], requires_grad=True
    )
    routing_rng_state = torch.random.get_rng_state()
    reference_output, reference_loss = model(source)
    (reference_output.square().mean() + reference_loss).backward()
    reference_input_gradient = source.grad.detach().clone()
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    source.grad = None
    torch.random.set_rng_state(routing_rng_state)
    if precision == torch.float64:
        model.double()
        source = source.detach().double().requires_grad_()
    with torch.autocast(
        "cpu", dtype=torch.bfloat16, enabled=precision == torch.bfloat16
    ):
        output, auxiliary_loss = model(source)
    loss_output = output.float() if precision == torch.bfloat16 else output
    (loss_output.square().mean() + auxiliary_loss).backward()
    if precision == torch.float64:
        assert output.dtype == torch.float64
        assert source.grad.dtype == torch.float64
    if logit_margins:
        assert min(logit_margins) > 0.05
    relative_tolerance, absolute_tolerance = (
        (0.08, 0.01) if precision == torch.bfloat16 else (1e-4, 1e-6)
    )
    torch.testing.assert_close(
        output,
        reference_output,
        rtol=0.05 if precision == torch.bfloat16 else relative_tolerance,
        atol=absolute_tolerance,
        check_dtype=False,
    )
    torch.testing.assert_close(
        source.grad,
        reference_input_gradient,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        check_dtype=False,
    )
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients.keys() == reference_gradients.keys(), {
        "mixed_only": gradients.keys() - reference_gradients.keys(),
        "reference_only": reference_gradients.keys() - gradients.keys(),
    }
    assert gradients
    for name, gradient in gradients.items():
        assert torch.isfinite(gradient).all(), name
        torch.testing.assert_close(
            gradient,
            reference_gradients[name],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            msg=name,
            check_dtype=False,
        )


@pytest.mark.parametrize("beam_width", [1, 2])
@pytest.mark.parametrize("tree_depth", [None, 2, 3])
def test_original_near_boundary_expert_initialization_stays_finite(
    beam_width, tree_depth
):
    model = _build_model(
        "expert_linear_adaptive",
        "NeuronExpertLinearAdaptiveConfigBuilder",
        beam_width,
        tree_depth,
    )
    source = torch.tensor(
        [[0.1, 0.3, -0.2, 0.4], [-0.1, 0.2, 0.3, -0.4]], requires_grad=True
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, auxiliary_loss = model(source)
    (output.float().square().mean() + auxiliary_loss).backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(auxiliary_loss).all()
    assert torch.isfinite(source.grad).all()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
