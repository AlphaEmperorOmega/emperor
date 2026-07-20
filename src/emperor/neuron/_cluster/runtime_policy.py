from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module


@dataclass(frozen=True)
class _ModulePolicy:
    module: Module
    training: bool
    role: str


@dataclass(frozen=True)
class _ParameterPolicy:
    parameter: Tensor
    requires_grad: bool
    device: torch.device
    dtype: torch.dtype
    role: str


@dataclass(frozen=True)
class _BufferPolicy:
    buffer: Tensor
    device: torch.device
    dtype: torch.dtype
    role: str


def inherit_runtime_policy(
    module: Module,
    template: Module,
    *,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype | None,
) -> None:
    """Apply the template's role-complete runtime policy to a fresh module.

    All tied-role conflicts are detected before the fresh module is mutated.
    Tensor contexts and trainability are applied before module modes so custom
    ``train()`` implementations observe their final parameter and buffer policy.
    """

    module_policies = _module_policies(module, template)
    parameter_policies = _parameter_policies(
        module,
        template,
        fallback_device=fallback_device,
        fallback_dtype=fallback_dtype,
    )
    buffer_policies = _buffer_policies(
        module,
        template,
        fallback_device=fallback_device,
        fallback_dtype=fallback_dtype,
    )

    for policy in parameter_policies.values():
        policy.parameter.data = policy.parameter.data.to(
            device=policy.device,
            dtype=policy.dtype,
        )
        policy.parameter.requires_grad_(policy.requires_grad)
    for policy in buffer_policies.values():
        policy.buffer.data = policy.buffer.data.to(
            device=policy.device,
            dtype=policy.dtype,
        )
    for policy in module_policies.values():
        policy.module.train(policy.training)


def _module_policies(
    module: Module,
    template: Module,
) -> dict[int, _ModulePolicy]:
    template_modules = dict(template.named_modules(remove_duplicate=False))
    policies: dict[int, _ModulePolicy] = {}
    for role, child in module.named_modules(remove_duplicate=False):
        template_child = template_modules.get(role)
        if template_child is None:
            continue
        existing_policy = policies.get(id(child))
        if (
            existing_policy is not None
            and existing_policy.training != template_child.training
        ):
            raise RuntimeError(
                "Cannot inherit Neuron runtime policy because grown module "
                f"roles {existing_policy.role!r} and {role!r} are tied but their "
                "template training modes differ."
            )
        policies[id(child)] = _ModulePolicy(
            module=child,
            training=template_child.training,
            role=role,
        )
    return policies


def _parameter_policies(
    module: Module,
    template: Module,
    *,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype | None,
) -> dict[int, _ParameterPolicy]:
    template_parameters = dict(template.named_parameters(remove_duplicate=False))
    policies: dict[int, _ParameterPolicy] = {}
    for role, parameter in module.named_parameters(remove_duplicate=False):
        template_parameter = template_parameters.get(role)
        device = (
            fallback_device if template_parameter is None else template_parameter.device
        )
        dtype = _inherited_tensor_dtype(
            parameter,
            template_parameter,
            fallback_dtype,
        )
        requires_grad = (
            parameter.requires_grad
            if template_parameter is None
            else template_parameter.requires_grad
        )
        existing_policy = policies.get(id(parameter))
        if (
            existing_policy is not None
            and existing_policy.requires_grad != requires_grad
        ):
            raise RuntimeError(
                "Cannot inherit Neuron runtime policy because grown parameter "
                f"roles {existing_policy.role!r} and {role!r} are tied but their "
                "template requires_grad policies differ."
            )
        if existing_policy is not None and (
            existing_policy.device,
            existing_policy.dtype,
        ) != (
            device,
            dtype,
        ):
            raise RuntimeError(
                "Cannot inherit Neuron runtime policy because grown parameter "
                f"roles {existing_policy.role!r} and {role!r} are tied but their "
                "template device or dtype contexts differ."
            )
        policies[id(parameter)] = _ParameterPolicy(
            parameter=parameter,
            requires_grad=requires_grad,
            device=device,
            dtype=dtype,
            role=role,
        )
    return policies


def _buffer_policies(
    module: Module,
    template: Module,
    *,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype | None,
) -> dict[int, _BufferPolicy]:
    template_buffers = dict(template.named_buffers(remove_duplicate=False))
    policies: dict[int, _BufferPolicy] = {}
    for role, buffer in module.named_buffers(remove_duplicate=False):
        template_buffer = template_buffers.get(role)
        device = fallback_device if template_buffer is None else template_buffer.device
        dtype = _inherited_tensor_dtype(
            buffer,
            template_buffer,
            fallback_dtype,
        )
        existing_policy = policies.get(id(buffer))
        if existing_policy is not None and (
            existing_policy.device,
            existing_policy.dtype,
        ) != (
            device,
            dtype,
        ):
            raise RuntimeError(
                "Cannot inherit Neuron runtime policy because grown buffer "
                f"roles {existing_policy.role!r} and {role!r} are tied but their "
                "template device or dtype contexts differ."
            )
        policies[id(buffer)] = _BufferPolicy(
            buffer=buffer,
            device=device,
            dtype=dtype,
            role=role,
        )
    return policies


def _inherited_tensor_dtype(
    tensor: Tensor,
    template_tensor: Tensor | None,
    fallback_dtype: torch.dtype | None,
) -> torch.dtype:
    if template_tensor is not None:
        return template_tensor.dtype
    if fallback_dtype is not None and (
        tensor.is_floating_point() or tensor.is_complex()
    ):
        return fallback_dtype
    return tensor.dtype
