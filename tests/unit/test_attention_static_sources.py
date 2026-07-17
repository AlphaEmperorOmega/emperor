import unittest
from dataclasses import replace

import torch
from torch import Tensor

from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
)
from support.attention import build_attention_config


def _set_stack_identity(stack) -> None:
    layer = stack.layers[0].model
    layer.weight_params.copy_(
        torch.eye(
            layer.weight_params.size(0),
            dtype=layer.weight_params.dtype,
            device=layer.weight_params.device,
        )
    )
    layer.bias_params.zero_()


def _reference_with_optional_static_sources(
    query: Tensor,
    dynamic_key: Tensor,
    dynamic_value: Tensor,
    static_key: Tensor | None,
    static_value: Tensor | None,
    *,
    num_heads: int,
) -> Tensor:
    target_length, batch_size, embedding_dim = query.shape
    source_length = dynamic_key.size(0)
    head_width = embedding_dim // num_heads
    query_heads = query.reshape(
        target_length,
        batch_size,
        num_heads,
        head_width,
    ).permute(1, 2, 0, 3)
    if static_key is None:
        key_heads = dynamic_key.reshape(
            source_length,
            batch_size,
            num_heads,
            head_width,
        ).permute(1, 2, 0, 3)
    else:
        key_heads = static_key.reshape(
            batch_size,
            num_heads,
            source_length,
            head_width,
        )
    if static_value is None:
        value_heads = dynamic_value.reshape(
            source_length,
            batch_size,
            num_heads,
            head_width,
        ).permute(1, 2, 0, 3)
    else:
        value_heads = static_value.reshape(
            batch_size,
            num_heads,
            source_length,
            head_width,
        )
    scores = torch.einsum("bhtd,bhsd->bhts", query_heads, key_heads)
    probabilities = torch.softmax(scores * head_width**-0.5, dim=-1)
    context = torch.einsum("bhts,bhsd->bhtd", probabilities, value_heads)
    return context.permute(2, 0, 1, 3).reshape(
        target_length,
        batch_size,
        embedding_dim,
    )


class TestStaticSourceSelectionAndGradients(unittest.TestCase):
    batch_size = 1
    num_heads = 1
    embedding_dim = 2
    target_length = 2
    source_length = 3

    def _build_model(self, config_class, *, configured_batch_size=None):
        options = {}
        if config_class is MixtureOfAttentionHeadsConfig:
            options = {
                "experts_top_k": 1,
                "experts_num_experts": 2,
                "experts_stack_num_layers": 1,
                "use_kv_expert_models_flag": False,
            }
        config = build_attention_config(
            config_class=config_class,
            batch_size=configured_batch_size or self.batch_size,
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            target_sequence_length=self.target_length,
            source_sequence_length=self.source_length,
            **options,
        )
        model = replace(
            config,
            target_dtype=torch.float64,
            batch_first_flag=False,
        ).build()
        model.eval()
        with torch.no_grad():
            if config_class is MixtureOfAttentionHeadsConfig:
                for parameter in model.parameters():
                    parameter.zero_()
                for expert in model.projector.query_model.expert_modules:
                    _set_stack_identity(expert)
                for expert in model.projector.output_model.expert_modules:
                    _set_stack_identity(expert)
                _set_stack_identity(model.projector.key_model)
                _set_stack_identity(model.projector.value_model)
            else:
                for projection in (
                    model.projector.query_model,
                    model.projector.key_model,
                    model.projector.value_model,
                    model.projector.output_model,
                ):
                    _set_stack_identity(projection)
        return model

    def _dynamic_inputs(self) -> tuple[Tensor, Tensor, Tensor]:
        query = torch.tensor(
            [[[1.0, -0.5]], [[2.0, 0.25]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        key = torch.tensor(
            [[[0.5, 1.0]], [[-1.0, 2.0]], [[3.0, -0.25]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        value = torch.tensor(
            [[[2.0, -1.0]], [[0.25, 4.0]], [[-3.0, 0.5]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        return query, key, value

    def _static_source(self, offset: float) -> Tensor:
        storage = torch.arange(
            self.batch_size
            * self.num_heads
            * self.source_length
            * self.embedding_dim
            * 2,
            dtype=torch.float64,
        ).reshape(
            self.batch_size * self.num_heads,
            self.source_length,
            self.embedding_dim * 2,
        )
        source = storage.div(7.0).add(offset)[..., ::2]
        self.assertFalse(source.is_contiguous())
        return source.detach().requires_grad_()

    def test_none_key_value_and_both_static_sources_match_values_and_gradients(
        self,
    ) -> None:
        cases = (
            ("dynamic", False, False),
            ("static_key", True, False),
            ("static_value", False, True),
            ("static_key_and_value", True, True),
        )
        for config_class in (
            IndependentAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        ):
            for name, use_static_key, use_static_value in cases:
                with self.subTest(config_class=config_class, case=name):
                    model = self._build_model(config_class)
                    query, dynamic_key, dynamic_value = self._dynamic_inputs()
                    static_key = self._static_source(0.75) if use_static_key else None
                    static_value = (
                        self._static_source(-1.25) if use_static_value else None
                    )
                    reference_query = query.detach().clone().requires_grad_()
                    reference_dynamic_key = (
                        dynamic_key.detach().clone().requires_grad_()
                    )
                    reference_dynamic_value = (
                        dynamic_value.detach().clone().requires_grad_()
                    )
                    reference_static_key = (
                        static_key.detach().clone().requires_grad_()
                        if static_key is not None
                        else None
                    )
                    reference_static_value = (
                        static_value.detach().clone().requires_grad_()
                        if static_value is not None
                        else None
                    )

                    output, weights, auxiliary_loss = model(
                        query,
                        dynamic_key,
                        dynamic_value,
                        static_k=static_key,
                        static_v=static_value,
                    )
                    reference_output = _reference_with_optional_static_sources(
                        reference_query,
                        reference_dynamic_key,
                        reference_dynamic_value,
                        reference_static_key,
                        reference_static_value,
                        num_heads=self.num_heads,
                    )
                    if config_class is MixtureOfAttentionHeadsConfig:
                        reference_output = reference_output * 0.5

                    torch.testing.assert_close(
                        output,
                        reference_output,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    self.assertIsNone(weights)
                    if config_class is IndependentAttentionConfig:
                        self.assertIsNone(auxiliary_loss)
                    else:
                        torch.testing.assert_close(
                            auxiliary_loss,
                            auxiliary_loss.new_zeros(()),
                        )

                    marker = torch.arange(
                        1,
                        output.numel() + 1,
                        dtype=output.dtype,
                    ).reshape_as(output)
                    objective = (output * marker).sum()
                    if auxiliary_loss is not None:
                        objective = objective + auxiliary_loss
                    objective.backward()
                    (reference_output * marker).sum().backward()

                    torch.testing.assert_close(
                        query.grad,
                        reference_query.grad,
                        rtol=1e-11,
                        atol=1e-11,
                    )
                    self._assert_selected_gradient(
                        dynamic_key,
                        reference_dynamic_key,
                        static_key,
                        reference_static_key,
                    )
                    self._assert_selected_gradient(
                        dynamic_value,
                        reference_dynamic_value,
                        static_value,
                        reference_static_value,
                    )

    def test_runtime_sized_static_sources_work_below_configured_batch_maximum(
        self,
    ) -> None:
        model = self._build_model(
            IndependentAttentionConfig,
            configured_batch_size=3,
        )
        query, dynamic_key, dynamic_value = self._dynamic_inputs()
        static_key = self._static_source(0.75)
        static_value = self._static_source(-1.25)
        reference_query = query.detach().clone().requires_grad_()
        reference_static_key = static_key.detach().clone().requires_grad_()
        reference_static_value = static_value.detach().clone().requires_grad_()

        output, weights, auxiliary_loss = model(
            query,
            dynamic_key,
            dynamic_value,
            static_k=static_key,
            static_v=static_value,
        )
        reference_output = _reference_with_optional_static_sources(
            reference_query,
            dynamic_key.detach(),
            dynamic_value.detach(),
            reference_static_key,
            reference_static_value,
            num_heads=self.num_heads,
        )

        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertIsNone(weights)
        self.assertIsNone(auxiliary_loss)

        marker = torch.arange(
            1,
            output.numel() + 1,
            dtype=output.dtype,
        ).reshape_as(output)
        (output * marker).sum().backward()
        (reference_output * marker).sum().backward()

        self.assertIsNone(dynamic_key.grad)
        self.assertIsNone(dynamic_value.grad)
        torch.testing.assert_close(query.grad, reference_query.grad)
        torch.testing.assert_close(static_key.grad, reference_static_key.grad)
        torch.testing.assert_close(static_value.grad, reference_static_value.grad)

    def test_mixture_runtime_static_sources_use_actual_not_configured_batch(
        self,
    ) -> None:
        model = self._build_model(
            MixtureOfAttentionHeadsConfig,
            configured_batch_size=3,
        )
        query, dynamic_key, dynamic_value = self._dynamic_inputs()
        static_key = self._static_source(0.75)
        static_value = self._static_source(-1.25)
        reference_query = query.detach().clone().requires_grad_()
        reference_static_key = static_key.detach().clone().requires_grad_()
        reference_static_value = static_value.detach().clone().requires_grad_()

        output, weights, auxiliary_loss = model(
            query,
            dynamic_key,
            dynamic_value,
            static_k=static_key,
            static_v=static_value,
        )
        reference_output = (
            _reference_with_optional_static_sources(
                reference_query,
                dynamic_key.detach(),
                dynamic_value.detach(),
                reference_static_key,
                reference_static_value,
                num_heads=self.num_heads,
            )
            * 0.5
        )

        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertIsNone(weights)
        torch.testing.assert_close(
            auxiliary_loss,
            auxiliary_loss.new_zeros(()),
        )

        marker = torch.arange(
            1,
            output.numel() + 1,
            dtype=output.dtype,
        ).reshape_as(output)
        ((output * marker).sum() + auxiliary_loss).backward()
        (reference_output * marker).sum().backward()

        self.assertIsNone(dynamic_key.grad)
        self.assertIsNone(dynamic_value.grad)
        torch.testing.assert_close(query.grad, reference_query.grad)
        torch.testing.assert_close(static_key.grad, reference_static_key.grad)
        torch.testing.assert_close(static_value.grad, reference_static_value.grad)
        self.assertTrue(torch.any(static_key.grad.abs() > 0))
        self.assertTrue(torch.any(static_value.grad.abs() > 0))

    def _assert_selected_gradient(
        self,
        dynamic_source: Tensor,
        reference_dynamic_source: Tensor,
        static_source: Tensor | None,
        reference_static_source: Tensor | None,
    ) -> None:
        if static_source is None:
            self.assertIsNotNone(dynamic_source.grad)
            self.assertTrue(torch.any(dynamic_source.grad.abs() > 0))
            torch.testing.assert_close(
                dynamic_source.grad,
                reference_dynamic_source.grad,
                rtol=1e-11,
                atol=1e-11,
            )
            return

        self.assertIsNone(dynamic_source.grad)
        self.assertIsNone(reference_dynamic_source.grad)
        self.assertIsNotNone(static_source.grad)
        self.assertTrue(torch.isfinite(static_source.grad).all())
        self.assertTrue(torch.any(static_source.grad.abs() > 0))
        torch.testing.assert_close(
            static_source.grad,
            reference_static_source.grad,
            rtol=1e-11,
            atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
