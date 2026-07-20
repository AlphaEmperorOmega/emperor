import unittest

import torch
from torch import Tensor, nn

from emperor.neuron import Terminal
from emperor.neuron._terminal_capture import (
    _anchor_private_scores_to_backward_hooks,
    _has_full_backward_hooks,
    _is_same_tensor_or_exact_view,
    _public_route_fields,
    _tensor_values_match,
)


class _DeterministicSampler(nn.Module):
    def sample_probabilities_log_scores_and_indices(self, input_matrix):
        probabilities = torch.softmax(input_matrix[:, :2], dim=-1)
        selected_indices = torch.tensor(
            [0, 1],
            dtype=torch.long,
            device=input_matrix.device,
        ).expand(input_matrix.shape[0], -1)
        return (
            probabilities,
            probabilities.log(),
            selected_indices,
            None,
            input_matrix.sum() * 0.0,
        )


def _build_terminal() -> Terminal:
    terminal = Terminal.__new__(Terminal)
    nn.Module.__init__(terminal)
    terminal.input_dim = 4
    terminal.sampler = _DeterministicSampler()
    terminal.register_buffer(
        "neuron_connections",
        torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.long),
        persistent=False,
    )
    return terminal


class _TerminalFactory:
    def build(self) -> Terminal:
        return _build_terminal()


class _TerminalCaptureTestCase(unittest.TestCase):
    batch_size = 2
    input_dim = 4

    def terminal_config(self) -> _TerminalFactory:
        return _TerminalFactory()


class TestTerminalCaptureContracts(_TerminalCaptureTestCase):
    def test_private_scores_require_public_autograd_edge_for_full_hook(self):
        model = self.terminal_config().build()
        hook_handle = model.register_full_backward_hook(
            lambda _module, grad_input, _grad_output: grad_input
        )
        differentiable_score = torch.tensor(1.0, requires_grad=True)
        detached_score = torch.tensor(2.0)
        public_output = tuple(torch.tensor(0.0) for _ in range(4))

        expected_message = (
            "Terminal scored routing cannot support a full backward hook when "
            "private scores require gradients but no public output carries an "
            "autograd edge. Attach the hook to the sampler/router instead."
        )
        try:
            with self.assertRaises(RuntimeError) as raised:
                _anchor_private_scores_to_backward_hooks(
                    model,
                    differentiable_score,
                    detached_score,
                    public_output,
                )
            self.assertEqual(str(raised.exception), expected_message)
            returned_scores = _anchor_private_scores_to_backward_hooks(
                model,
                detached_score,
                detached_score,
                public_output,
            )
        finally:
            hook_handle.remove()

        self.assertIs(returned_scores[0], detached_score)
        self.assertIs(returned_scores[1], detached_score)

    def test_private_score_anchor_preserves_values_and_exact_cotangents(self):
        model = self.terminal_config().build()
        hook_handle = model.register_full_backward_hook(
            lambda _module, grad_input, _grad_output: grad_input
        )
        log_probabilities = torch.tensor(
            [1.0, 2.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        router_scores = torch.tensor(
            [3.0, 4.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        public_anchor = torch.tensor(
            [5.0, 6.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        public_output = (
            public_anchor,
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

        try:
            anchored_logs, anchored_router_scores = (
                _anchor_private_scores_to_backward_hooks(
                    model,
                    log_probabilities,
                    router_scores,
                    public_output,
                )
            )
        finally:
            hook_handle.remove()

        self.assertIsNot(anchored_logs, log_probabilities)
        self.assertIsNot(anchored_router_scores, router_scores)
        torch.testing.assert_close(anchored_logs, log_probabilities)
        torch.testing.assert_close(anchored_router_scores, router_scores)

        (anchored_logs.sum() + 2.0 * anchored_router_scores.sum()).backward()

        torch.testing.assert_close(
            log_probabilities.grad,
            torch.ones_like(log_probabilities),
        )
        torch.testing.assert_close(
            router_scores.grad,
            torch.full_like(router_scores, 2.0),
        )
        torch.testing.assert_close(public_anchor.grad, torch.zeros_like(public_anchor))

    def test_full_backward_hook_detection_covers_local_and_global_hooks(self):
        module_hooks = torch.nn.modules.module
        model = self.terminal_config().build()
        self.assertFalse(_has_full_backward_hooks(model))

        local_handle = model.register_full_backward_hook(
            lambda _module, grad_input, _grad_output: grad_input
        )
        try:
            self.assertTrue(_has_full_backward_hooks(model))
        finally:
            local_handle.remove()

        original_global_hook_kind = module_hooks._global_is_full_backward_hook
        global_handle = module_hooks.register_module_full_backward_hook(
            lambda _module, grad_input, _grad_output: grad_input
        )
        try:
            self.assertTrue(_has_full_backward_hooks(model))
        finally:
            global_handle.remove()
            module_hooks._global_is_full_backward_hook = original_global_hook_kind

    def test_log_probability_seam_preserves_public_route_fields(self):
        model = self.terminal_config().build().double()
        input_batch = torch.randn(
            self.batch_size,
            self.input_dim,
            dtype=torch.float64,
        )

        routed_signal = model._forward_with_log_probabilities(input_batch)

        self.assertEqual(len(routed_signal), 5)
        self.assertIs(routed_signal[0], input_batch)
        torch.testing.assert_close(routed_signal[2], routed_signal[1].log())
        self.assertEqual(routed_signal[3].shape[-1], 3)

    def test_legacy_five_field_sampler_uses_log_probabilities_as_router_scores(self):
        class FiveFieldSampler(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def sample_probabilities_log_scores_and_indices(self, input_matrix):
                self.call_count += 1
                probabilities = torch.softmax(input_matrix[:, :2], dim=-1)
                log_probabilities = probabilities.log()
                indices = torch.tensor(
                    [0, 1],
                    dtype=torch.long,
                    device=input_matrix.device,
                ).expand(input_matrix.shape[0], -1)
                return (
                    probabilities,
                    log_probabilities,
                    indices,
                    None,
                    input_matrix.sum() * 0.0,
                )

        model = self.terminal_config().build().double()
        sampler = FiveFieldSampler()
        model.sampler = sampler
        input_batch = torch.randn(
            self.batch_size,
            self.input_dim,
            dtype=torch.float64,
            requires_grad=True,
        )

        routed_signal = model._forward_with_router_scores(input_batch)

        self.assertEqual(sampler.call_count, 1)
        self.assertIs(routed_signal[2], routed_signal[3])
        routed_signal[3].sum().backward()
        self.assertTrue(torch.isfinite(input_batch.grad).all().item())

    def test_scored_forward_requires_terminal_to_publish_route(self):
        class NonPublishingTerminal(Terminal):
            def forward(self, input: Tensor):
                probabilities = input.new_ones((input.shape[0], 1))
                selected_neurons = torch.zeros(
                    input.shape[0],
                    1,
                    3,
                    dtype=torch.long,
                    device=input.device,
                )
                return input, probabilities, selected_neurons, input.new_zeros(())

        model = NonPublishingTerminal.__new__(NonPublishingTerminal)
        nn.Module.__init__(model)
        model.input_dim = self.input_dim

        with self.assertRaisesRegex(RuntimeError, "did not publish"):
            model._forward_with_router_scores(
                torch.randn(self.batch_size, self.input_dim)
            )

    def test_nested_capture_aligns_by_unchanged_scored_fields(self):
        model = self.terminal_config().build()
        outer_input = torch.randn(self.batch_size, self.input_dim)
        nested_input = torch.randn_like(outer_input)
        nested_call_active = False

        def nest_then_replace_unscored_fields(module, _inputs, output):
            nonlocal nested_call_active
            if nested_call_active:
                return None
            nested_call_active = True
            try:
                module(nested_input)
            finally:
                nested_call_active = False
            input, probabilities, selected_neurons, auxiliary_loss = output
            return (
                input + 1.0,
                probabilities,
                selected_neurons,
                auxiliary_loss + 2.0,
            )

        hook_handle = model.register_forward_hook(nest_then_replace_unscored_fields)
        try:
            routed_signal = model._forward_with_router_scores(outer_input)
        finally:
            hook_handle.remove()

        torch.testing.assert_close(routed_signal[0], outer_input + 1.0)
        torch.testing.assert_close(routed_signal[5], outer_input.new_tensor(2.0))

    def test_nested_capture_uses_requested_input_before_rejecting_replaced_scores(self):
        model = self.terminal_config().build()
        outer_input = torch.randn(self.batch_size, self.input_dim)
        nested_input = torch.randn_like(outer_input)
        nested_call_active = False

        def nest_then_replace_scores(module, _inputs, output):
            nonlocal nested_call_active
            if nested_call_active:
                return None
            nested_call_active = True
            try:
                module(nested_input)
            finally:
                nested_call_active = False
            input, probabilities, selected_neurons, auxiliary_loss = output
            return (
                input + 1.0,
                probabilities.clone(),
                selected_neurons.clone(),
                auxiliary_loss,
            )

        hook_handle = model.register_forward_hook(nest_then_replace_scores)
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "replaced scored routing output tensors",
            ):
                model._forward_with_router_scores(outer_input)
        finally:
            hook_handle.remove()

    def test_nested_capture_rejects_ambiguous_same_input_records(self):
        model = self.terminal_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)
        nested_call_active = False

        def nest_same_input_then_replace_scores(module, inputs, output):
            nonlocal nested_call_active
            if nested_call_active:
                return None
            nested_call_active = True
            try:
                module(inputs[0])
            finally:
                nested_call_active = False
            input, probabilities, selected_neurons, auxiliary_loss = output
            return (
                input + 1.0,
                probabilities.clone(),
                selected_neurons.clone(),
                auxiliary_loss,
            )

        hook_handle = model.register_forward_hook(nest_same_input_then_replace_scores)
        try:
            with self.assertRaisesRegex(RuntimeError, "could not uniquely align"):
                model._forward_with_router_scores(input_batch)
        finally:
            hook_handle.remove()

    def test_scored_forward_detects_shape_mutation_through_tensor_data(self):
        model = self.terminal_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        def flatten_selected_neurons(_module, _inputs, output):
            output[2].data = output[2].data.reshape(-1)

        hook_handle = model.register_forward_hook(flatten_selected_neurons)
        try:
            with self.assertRaisesRegex(RuntimeError, "mutated.*selected_neurons"):
                model._forward_with_router_scores(input_batch)
        finally:
            hook_handle.remove()

    def test_scored_forward_rejects_disconnected_same_storage_probability(self):
        model = self.terminal_config().build().double()
        input_batch = torch.randn(
            self.batch_size,
            self.input_dim,
            dtype=torch.float64,
            requires_grad=True,
        )

        def disconnect_probability_graph(_module, _inputs, output):
            input, probabilities, selected_neurons, auxiliary_loss = output
            return (
                input,
                probabilities.detach().requires_grad_(),
                selected_neurons,
                auxiliary_loss,
            )

        hook_handle = model.register_forward_hook(disconnect_probability_graph)
        try:
            with self.assertRaisesRegex(RuntimeError, "replaced.*probabilities"):
                model._forward_with_router_scores(input_batch)
        finally:
            hook_handle.remove()

    def test_exact_view_predicate_rejects_non_tensor_values(self):
        self.assertFalse(_is_same_tensor_or_exact_view(torch.ones(1), object()))

    def test_public_route_fields_preserve_exact_public_tensor_identities(self):
        route_fields = tuple(torch.tensor(float(index)) for index in range(6))

        public_fields = _public_route_fields(route_fields)

        self.assertEqual(len(public_fields), 4)
        for public_field, route_index in zip(
            public_fields,
            (0, 1, 4, 5),
            strict=True,
        ):
            self.assertIs(public_field, route_fields[route_index])

    def test_tensor_snapshot_comparison_requires_exact_metadata_and_nan_layout(self):
        cases = (
            (
                "dtype-mismatch",
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float64),
                False,
            ),
            (
                "broadcastable-shape-mismatch",
                torch.tensor([1.0]),
                torch.tensor([[1.0]]),
                False,
            ),
            (
                "matching-nan-layout",
                torch.tensor([1.0, float("nan")]),
                torch.tensor([1.0, float("nan")]),
                True,
            ),
            (
                "different-nan-layout",
                torch.tensor([float("nan"), 1.0]),
                torch.tensor([1.0, float("nan")]),
                False,
            ),
            (
                "matching-complex-nan-layout",
                torch.tensor([complex(float("nan"), 1.0)]),
                torch.tensor([complex(float("nan"), 1.0)]),
                True,
            ),
        )
        for case_name, snapshot, current, expected in cases:
            with self.subTest(case_name=case_name):
                self.assertIs(_tensor_values_match(snapshot, current), expected)


if __name__ == "__main__":
    unittest.main()
