from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from torch import Tensor

from emperor.nn import Module

TerminalRoute: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
TerminalPublicOutput: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class _TerminalRouteRecord:
    routed_signal: TerminalRoute
    scored_field_versions: tuple[int | None, int | None]
    scored_field_snapshots: tuple[Tensor | None, Tensor | None]
    probability_graph_metadata: tuple[bool, object | None, bool]


@dataclass
class _TerminalRouteCapture:
    owner: object | None
    records: list[_TerminalRouteRecord] = field(default_factory=list)


_TERMINAL_ROUTE_CAPTURE: ContextVar[_TerminalRouteCapture | None] = ContextVar(
    "emperor_neuron_terminal_route_capture",
    default=None,
)


class _PrivateScoreBackwardHookAnchor(torch.autograd.Function):
    """Activate Terminal output hooks without changing private score values."""

    @staticmethod
    def forward(ctx, private_score: Tensor, public_anchor: Tensor) -> Tensor:
        ctx.save_for_backward(public_anchor)
        return private_score

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, Tensor]:
        (public_anchor,) = ctx.saved_tensors
        return gradient, torch.zeros_like(public_anchor)


def publish_terminal_route(owner: Module, routed_signal: TerminalRoute) -> None:
    active_route_capture = _TERMINAL_ROUTE_CAPTURE.get()
    if active_route_capture is None or active_route_capture.owner is not owner:
        return

    probabilities = routed_signal[1]
    selected_neurons = routed_signal[4]
    should_snapshot_scored_fields = _forward_hooks_may_mutate_output(owner)
    active_route_capture.records.append(
        _TerminalRouteRecord(
            routed_signal=routed_signal,
            scored_field_versions=(
                _tensor_version(probabilities),
                _tensor_version(selected_neurons),
            ),
            scored_field_snapshots=(
                probabilities.detach().clone()
                if should_snapshot_scored_fields
                else None,
                selected_neurons.detach().clone()
                if should_snapshot_scored_fields
                else None,
            ),
            probability_graph_metadata=(
                probabilities.requires_grad,
                probabilities.grad_fn,
                probabilities.is_leaf,
            ),
        )
    )


def run_scored_terminal_forward(owner: Module, input: Tensor) -> TerminalRoute:
    _validate_scored_hook_capabilities(owner)
    route_capture = _TerminalRouteCapture(owner=owner)
    route_capture_token = _TERMINAL_ROUTE_CAPTURE.set(route_capture)
    try:
        hooked_public_output = owner(input)
        captured_records = tuple(route_capture.records)
    finally:
        _TERMINAL_ROUTE_CAPTURE.reset(route_capture_token)
        route_capture.owner = None
        route_capture.records.clear()

    public_output_fields = _coerce_public_output(hooked_public_output)
    captured_record = _select_captured_record(
        captured_records,
        requested_input=input,
        public_output=public_output_fields,
    )
    routed_signal = captured_record.routed_signal
    _validate_scored_output(captured_record, public_output_fields)
    routed_input, probabilities, selected_neurons, auxiliary_loss = public_output_fields
    _, _, log_probabilities, router_scores, _, _ = routed_signal
    log_probabilities, router_scores = _anchor_private_scores_to_backward_hooks(
        owner,
        log_probabilities,
        router_scores,
        public_output_fields,
    )
    return (
        routed_input,
        probabilities,
        log_probabilities,
        router_scores,
        selected_neurons,
        auxiliary_loss,
    )


def _anchor_private_scores_to_backward_hooks(
    owner: Module,
    log_probabilities: Tensor,
    router_scores: Tensor,
    public_output: TerminalPublicOutput,
) -> tuple[Tensor, Tensor]:
    if not _has_full_backward_hooks(owner):
        return log_probabilities, router_scores
    public_anchor = next(
        (output_field for output_field in public_output if output_field.requires_grad),
        None,
    )
    private_scores_require_grad = (
        log_probabilities.requires_grad or router_scores.requires_grad
    )
    if public_anchor is None:
        if private_scores_require_grad:
            raise RuntimeError(
                "Terminal scored routing cannot support a full backward hook when "
                "private scores require gradients but no public output carries an "
                "autograd edge. Attach the hook to the sampler/router instead."
            )
        return log_probabilities, router_scores

    anchored_log_probabilities = _PrivateScoreBackwardHookAnchor.apply(
        log_probabilities,
        public_anchor,
    )
    anchored_router_scores = (
        anchored_log_probabilities
        if router_scores is log_probabilities
        else _PrivateScoreBackwardHookAnchor.apply(router_scores, public_anchor)
    )
    return anchored_log_probabilities, anchored_router_scores


def _validate_scored_hook_capabilities(module: Module) -> None:
    if _has_full_backward_pre_hooks(module):
        raise RuntimeError(
            "Terminal scored routing does not support full backward pre-hooks: "
            "private log/router-score cotangents are not represented by the "
            "public four-field Terminal output. Attach gradient instrumentation "
            "to the sampler/router instead."
        )
    if _has_legacy_backward_hooks(module):
        raise RuntimeError(
            "Terminal scored routing does not support legacy backward hooks: "
            "they cannot observe gradients from private score outputs. Use a "
            "full backward hook or attach instrumentation to the sampler/router."
        )


def _coerce_public_output(public_output: object) -> TerminalPublicOutput:
    try:
        public_output_fields = tuple(public_output)  # type: ignore[arg-type]
    except TypeError as output_type_error:
        raise RuntimeError(
            "Terminal forward hook replaced the four-field output during scored "
            "routing; private log/router scores cannot be aligned."
        ) from output_type_error
    if len(public_output_fields) != 4:
        raise RuntimeError(
            "Terminal forward hook replaced the four-field output during scored "
            "routing; private log/router scores cannot be aligned."
        )
    if not all(
        isinstance(output_field, Tensor) for output_field in public_output_fields
    ):
        raise RuntimeError(
            "Terminal forward hook replaced a Tensor in the four-field output "
            "during scored routing; private log/router scores cannot be aligned."
        )
    return (
        public_output_fields[0],
        public_output_fields[1],
        public_output_fields[2],
        public_output_fields[3],
    )


def _validate_scored_output(
    record: _TerminalRouteRecord,
    public_output: TerminalPublicOutput,
) -> None:
    captured_route = record.routed_signal
    captured_scored_fields = (captured_route[1], captured_route[4])
    public_scored_fields = (public_output[1], public_output[2])
    scored_field_names = ("probabilities", "selected_neurons")
    replaced_scored_fields = [
        field_name
        for field_name, captured_field, public_field in zip(
            scored_field_names,
            captured_scored_fields,
            public_scored_fields,
            strict=True,
        )
        if not _is_same_tensor_or_exact_view(captured_field, public_field)
    ]
    if replaced_scored_fields:
        replaced_field_names = ", ".join(replaced_scored_fields)
        raise RuntimeError(
            "Terminal forward hook replaced scored routing output tensors "
            f"({replaced_field_names}); private log/router scores would be "
            "misaligned."
        )

    public_probabilities = public_scored_fields[0]
    public_selected_neurons = public_scored_fields[1]
    mutated_scored_fields = [
        field_name
        for field_name, public_field, captured_version, captured_snapshot in zip(
            scored_field_names,
            (public_probabilities, public_selected_neurons),
            record.scored_field_versions,
            record.scored_field_snapshots,
            strict=True,
        )
        if (captured_version is not None and public_field._version != captured_version)
        or (
            captured_snapshot is not None
            and not _tensor_values_match(captured_snapshot, public_field)
        )
    ]
    captured_probability = captured_scored_fields[0]
    if _has_probability_backward_observer(
        captured_probability,
        public_probabilities,
    ):
        raise RuntimeError(
            "Terminal scored routing does not support probability Tensor backward "
            "hooks or retain_grad(): private log/router-score cotangents are not "
            "represented by the public probability output. Attach instrumentation "
            "to the sampler/router instead."
        )
    captured_probability_graph_metadata = (
        captured_probability.requires_grad,
        captured_probability.grad_fn,
        captured_probability.is_leaf,
    )
    probability_graph_changed = (
        captured_probability_graph_metadata[0] != record.probability_graph_metadata[0]
        or captured_probability_graph_metadata[1]
        is not record.probability_graph_metadata[1]
        or captured_probability_graph_metadata[2]
        != record.probability_graph_metadata[2]
    )
    if public_probabilities is not captured_probability:
        probability_graph_changed |= (
            public_probabilities.requires_grad != captured_probability.requires_grad
        )
    if probability_graph_changed:
        mutated_scored_fields.append("probabilities")
    if mutated_scored_fields:
        mutated_field_names = ", ".join(dict.fromkeys(mutated_scored_fields))
        raise RuntimeError(
            "Terminal forward hook mutated scored routing output tensors "
            f"({mutated_field_names}); private log/router scores would be misaligned."
        )


def _has_probability_backward_observer(
    captured_probability: Tensor,
    public_probability: Tensor,
) -> bool:
    return any(
        bool(getattr(probability, "_backward_hooks", None)) or probability.retains_grad
        for probability in (captured_probability, public_probability)
    )


def _select_captured_record(
    captured_records: tuple[_TerminalRouteRecord, ...],
    *,
    requested_input: Tensor,
    public_output: TerminalPublicOutput,
) -> _TerminalRouteRecord:
    if not captured_records:
        raise RuntimeError("Terminal forward did not publish its routed signal.")
    if len(captured_records) == 1:
        return captured_records[0]

    exact_output_records = [
        captured_record
        for captured_record in captured_records
        if all(
            _is_same_tensor_or_exact_view(captured_field, public_field)
            for captured_field, public_field in zip(
                _public_route_fields(captured_record.routed_signal),
                public_output,
                strict=True,
            )
        )
    ]
    if len(exact_output_records) == 1:
        return exact_output_records[0]

    scored_output_records = [
        captured_record
        for captured_record in captured_records
        if _is_same_tensor_or_exact_view(
            captured_record.routed_signal[1],
            public_output[1],
        )
        and _is_same_tensor_or_exact_view(
            captured_record.routed_signal[4],
            public_output[2],
        )
    ]
    if len(scored_output_records) == 1:
        return scored_output_records[0]

    requested_input_records = [
        captured_record
        for captured_record in captured_records
        if _is_same_tensor_or_exact_view(
            requested_input,
            captured_record.routed_signal[0],
        )
    ]
    if len(requested_input_records) == 1:
        return requested_input_records[0]
    raise RuntimeError(
        "Terminal scored routing could not uniquely align its forward result with "
        "a captured invocation."
    )


def _public_route_fields(routed_signal: TerminalRoute) -> TerminalPublicOutput:
    return (
        routed_signal[0],
        routed_signal[1],
        routed_signal[4],
        routed_signal[5],
    )


def _tensor_version(tensor: Tensor) -> int | None:
    if torch.is_inference(tensor):
        return None
    return tensor._version


def _forward_hooks_may_mutate_output(module: Module) -> bool:
    global_forward_hooks = getattr(
        torch.nn.modules.module,
        "_global_forward_hooks",
        (),
    )
    return bool(module._forward_hooks) or bool(global_forward_hooks)


def _has_full_backward_pre_hooks(module: Module) -> bool:
    global_backward_pre_hooks = getattr(
        torch.nn.modules.module,
        "_global_backward_pre_hooks",
        (),
    )
    return bool(module._backward_pre_hooks) or bool(global_backward_pre_hooks)


def _has_full_backward_hooks(module: Module) -> bool:
    global_backward_hooks = getattr(
        torch.nn.modules.module,
        "_global_backward_hooks",
        (),
    )
    global_hook_kind = getattr(
        torch.nn.modules.module,
        "_global_is_full_backward_hook",
        None,
    )
    has_local_full_hook = (
        bool(module._backward_hooks) and module._is_full_backward_hook is True
    )
    has_global_full_hook = bool(global_backward_hooks) and global_hook_kind is True
    return has_local_full_hook or has_global_full_hook


def _has_legacy_backward_hooks(module: Module) -> bool:
    global_backward_hooks = getattr(
        torch.nn.modules.module,
        "_global_backward_hooks",
        (),
    )
    global_hook_kind = getattr(
        torch.nn.modules.module,
        "_global_is_full_backward_hook",
        None,
    )
    has_local_legacy_hook = (
        bool(module._backward_hooks) and module._is_full_backward_hook is False
    )
    has_global_legacy_hook = bool(global_backward_hooks) and global_hook_kind is False
    return has_local_legacy_hook or has_global_legacy_hook


def _tensor_values_match(snapshot: Tensor, current: Tensor) -> bool:
    if (
        snapshot.shape != current.shape
        or snapshot.dtype != current.dtype
        or snapshot.device != current.device
    ):
        return False
    if torch.equal(snapshot, current):
        return True
    if snapshot.dtype.is_floating_point or snapshot.dtype.is_complex:
        equal_or_matching_nan = (snapshot == current) | (
            torch.isnan(snapshot) & torch.isnan(current)
        )
        return bool(equal_or_matching_nan.all().item())
    return False


def _is_same_tensor_or_exact_view(captured: object, public: object) -> bool:
    if public is captured:
        return True
    if not isinstance(captured, Tensor) or not isinstance(public, Tensor):
        return False
    if public.requires_grad != captured.requires_grad or not public.is_set_to(captured):
        return False
    if not captured.requires_grad:
        return True
    public_gradient_node = torch.autograd.graph.get_gradient_edge(public).node
    captured_gradient_node = torch.autograd.graph.get_gradient_edge(captured).node
    pending_gradient_nodes = [public_gradient_node]
    visited_gradient_node_ids: set[int] = set()
    while pending_gradient_nodes:
        gradient_node = pending_gradient_nodes.pop()
        if gradient_node is captured_gradient_node:
            return True
        gradient_node_id = id(gradient_node)
        if gradient_node_id in visited_gradient_node_ids:
            continue
        visited_gradient_node_ids.add(gradient_node_id)
        for next_gradient_node, _ in gradient_node.next_functions:
            if next_gradient_node is not None:
                pending_gradient_nodes.append(next_gradient_node)
    return False
