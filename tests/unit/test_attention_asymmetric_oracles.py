import unittest
from dataclasses import replace

import torch
from torch import Tensor

from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.embedding.relative import DynamicPositionalBiasConfig
from support.attention import build_attention_config


def _reference_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    num_heads: int,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    target_length, batch_size, embedding_dim = query.shape
    source_length = key.size(0)
    head_width = embedding_dim // num_heads
    query_heads = query.reshape(
        target_length,
        batch_size,
        num_heads,
        head_width,
    ).permute(1, 2, 0, 3)
    key_heads = key.reshape(
        source_length,
        batch_size,
        num_heads,
        head_width,
    ).permute(1, 2, 0, 3)
    value_heads = value.reshape(
        source_length,
        batch_size,
        num_heads,
        head_width,
    ).permute(1, 2, 0, 3)
    scores = torch.einsum("bhtd,bhsd->bhts", query_heads, key_heads)
    scaled_scores = scores * head_width**-0.5
    if attention_mask is not None:
        scaled_scores = scaled_scores + attention_mask
    probabilities = torch.softmax(scaled_scores, dim=-1)
    context = torch.einsum("bhts,bhsd->bhtd", probabilities, value_heads)
    output = context.permute(2, 0, 1, 3).reshape(
        target_length,
        batch_size,
        embedding_dim,
    )
    return output, probabilities


def _reference_attention_with_relative_bias(
    inputs: Tensor,
    relative_table: Tensor,
    *,
    num_heads: int,
    max_positions: int,
) -> tuple[Tensor, Tensor]:
    target_length, batch_size, embedding_dim = inputs.shape
    head_width = embedding_dim // num_heads
    heads = inputs.reshape(
        target_length,
        batch_size,
        num_heads,
        head_width,
    ).permute(1, 2, 0, 3)
    scaled_query = heads * head_width**-0.5
    content_scores = torch.einsum("bhtd,bhsd->bhts", scaled_query, heads)
    relative_indices = (
        torch.arange(target_length, device=inputs.device)[None, :]
        - torch.arange(target_length, device=inputs.device)[:, None]
        + max_positions
    )
    relative_vectors = relative_table[:, :, relative_indices]
    relative_scores = torch.einsum(
        "bhtd,hdts->bhts",
        scaled_query,
        relative_vectors,
    )
    probabilities = torch.softmax(content_scores + relative_scores, dim=-1)
    context = torch.einsum("bhts,bhsd->bhtd", probabilities, heads)
    output = context.permute(2, 0, 1, 3).reshape(
        target_length,
        batch_size,
        embedding_dim,
    )
    return output, probabilities


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


class TestAsymmetricStandardAttentionOracle(unittest.TestCase):
    batch_size = 2
    num_heads = 3
    embedding_dim = 6
    target_length = 5
    source_length = 6

    def _inputs(self) -> tuple[Tensor, Tensor, Tensor]:
        query = (
            torch.arange(
                self.target_length * self.batch_size * self.embedding_dim,
                dtype=torch.float64,
            )
            .reshape(self.target_length, self.batch_size, self.embedding_dim)
            .div_(17.0)
            .sub_(1.5)
            .requires_grad_()
        )
        key = (
            torch.arange(
                self.source_length * self.batch_size * self.embedding_dim,
                dtype=torch.float64,
            )
            .flip(0)
            .reshape(self.source_length, self.batch_size, self.embedding_dim)
            .div_(19.0)
            .sub_(1.0)
            .requires_grad_()
        )
        value = (
            torch.arange(
                self.source_length * self.batch_size * self.embedding_dim,
                dtype=torch.float64,
            )
            .roll(7)
            .reshape(self.source_length, self.batch_size, self.embedding_dim)
            .div_(13.0)
            .sub_(2.0)
            .requires_grad_()
        )
        return query, key, value

    def _config(self, config_class):
        source_length = (
            self.target_length
            if config_class is SelfAttentionConfig
            else self.source_length
        )
        config = build_attention_config(
            config_class=config_class,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            target_sequence_length=self.target_length,
            source_sequence_length=source_length,
            return_attention_weights_flag=(config_class is SelfAttentionConfig),
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        return replace(
            config,
            target_dtype=torch.float64,
            batch_first_flag=False,
        )

    def test_self_and_independent_match_exact_multi_axis_reference_and_gradients(
        self,
    ) -> None:
        for config_class in (SelfAttentionConfig, IndependentAttentionConfig):
            with self.subTest(config_class=config_class):
                model = self._config(config_class).build().eval()
                with torch.no_grad():
                    for projection in (
                        model.projector.query_model,
                        model.projector.key_model,
                        model.projector.value_model,
                        model.projector.output_model,
                    ):
                        _set_stack_identity(projection)

                query, key, value = self._inputs()
                if config_class is SelfAttentionConfig:
                    key = value = query
                reference_query = query.detach().clone().requires_grad_()
                if config_class is SelfAttentionConfig:
                    reference_key = reference_value = reference_query
                else:
                    reference_key = key.detach().clone().requires_grad_()
                    reference_value = value.detach().clone().requires_grad_()

                output, weights, auxiliary_loss = model(query, key, value)
                reference_output, reference_weights = _reference_attention(
                    reference_query,
                    reference_key,
                    reference_value,
                    num_heads=self.num_heads,
                )

                torch.testing.assert_close(
                    output,
                    reference_output,
                    rtol=1e-12,
                    atol=1e-12,
                )
                if config_class is SelfAttentionConfig:
                    torch.testing.assert_close(
                        weights,
                        reference_weights,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                else:
                    self.assertIsNone(weights)
                self.assertIsNone(auxiliary_loss)

                marker = torch.arange(
                    1,
                    output.numel() + 1,
                    dtype=output.dtype,
                ).reshape_as(output)
                (output * marker).sum().backward()
                (reference_output * marker).sum().backward()
                torch.testing.assert_close(
                    query.grad,
                    reference_query.grad,
                    rtol=1e-11,
                    atol=1e-11,
                )
                if config_class is IndependentAttentionConfig:
                    torch.testing.assert_close(
                        key.grad,
                        reference_key.grad,
                        rtol=1e-11,
                        atol=1e-11,
                    )
                    torch.testing.assert_close(
                        value.grad,
                        reference_value.grad,
                        rtol=1e-11,
                        atol=1e-11,
                    )


class TestCausalMaskCompositionOracle(unittest.TestCase):
    sequence_length = 4
    batch_size = 2
    num_heads = 2
    embedding_dim = 4

    def _model(self, batch_first_flag: bool):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            target_sequence_length=self.sequence_length,
            source_sequence_length=self.sequence_length,
            causal_attention_mask_flag=True,
            return_attention_weights_flag=True,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        model = replace(
            config,
            target_dtype=torch.float64,
            batch_first_flag=batch_first_flag,
        ).build()
        model.eval()
        with torch.no_grad():
            for projection in (
                model.projector.query_model,
                model.projector.key_model,
                model.projector.value_model,
                model.projector.output_model,
            ):
                _set_stack_identity(projection)
        return model

    def _sequence_values(self) -> Tensor:
        return (
            torch.arange(
                self.sequence_length * self.batch_size * self.embedding_dim,
                dtype=torch.float64,
            )
            .reshape(self.sequence_length, self.batch_size, self.embedding_dim)
            .roll(5, dims=0)
            .div_(11.0)
            .sub_(1.25)
        )

    def test_explicit_additive_mask_composes_with_causality_across_layouts(self):
        explicit_mask = torch.tensor(
            (
                (0.00, 4.00, 5.00, 6.00),
                (0.35, -0.20, 7.00, 8.00),
                (0.50, 0.10, -0.40, 9.00),
                (0.70, -0.10, 0.20, 0.00),
            ),
            dtype=torch.float64,
        )
        causal_positions = torch.triu(
            torch.ones(
                self.sequence_length,
                self.sequence_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        composed_mask = explicit_mask.masked_fill(causal_positions, -torch.inf)

        for layout in ("sequence_first", "batch_first", "unbatched"):
            with self.subTest(layout=layout):
                sequence_values = self._sequence_values()
                reference_inputs = sequence_values
                batch_first_flag = layout == "batch_first"
                if layout == "unbatched":
                    reference_inputs = sequence_values[:, :1]
                    model_inputs = sequence_values[:, 0]
                elif batch_first_flag:
                    model_inputs = sequence_values.transpose(0, 1).contiguous()
                else:
                    model_inputs = sequence_values
                model_inputs = model_inputs.clone().requires_grad_()
                reference_inputs = reference_inputs.clone().requires_grad_()
                model = self._model(batch_first_flag)

                output, weights, auxiliary_loss = model(
                    model_inputs,
                    model_inputs,
                    model_inputs,
                    attention_mask=explicit_mask,
                )
                reference_output, reference_weights = _reference_attention(
                    reference_inputs,
                    reference_inputs,
                    reference_inputs,
                    num_heads=self.num_heads,
                    attention_mask=composed_mask,
                )
                if layout == "unbatched":
                    reference_output = reference_output.squeeze(1)
                    reference_weights = reference_weights.squeeze(0)
                elif batch_first_flag:
                    reference_output = reference_output.transpose(0, 1).contiguous()

                torch.testing.assert_close(
                    output,
                    reference_output,
                    rtol=1e-12,
                    atol=1e-12,
                )
                torch.testing.assert_close(
                    weights,
                    reference_weights,
                    rtol=1e-12,
                    atol=1e-12,
                )
                self.assertIsNone(auxiliary_loss)

                marker = torch.arange(
                    1,
                    output.numel() + 1,
                    dtype=output.dtype,
                ).reshape_as(output)
                (output * marker).sum().backward()
                (reference_output * marker).sum().backward()
                expected_input_gradient = reference_inputs.grad
                if layout == "unbatched":
                    expected_input_gradient = expected_input_gradient.squeeze(1)
                elif batch_first_flag:
                    expected_input_gradient = expected_input_gradient.transpose(
                        0,
                        1,
                    ).contiguous()
                torch.testing.assert_close(
                    model_inputs.grad,
                    expected_input_gradient,
                    rtol=1e-11,
                    atol=1e-11,
                )

    def test_explicit_zero_mask_cannot_leak_future_values_or_gradients(self):
        cutoff = 2
        explicit_mask = torch.zeros(
            self.sequence_length,
            self.sequence_length,
            dtype=torch.float64,
        )

        for layout in ("sequence_first", "batch_first", "unbatched"):
            with self.subTest(layout=layout):
                base_sequence = self._sequence_values()
                changed_sequence = base_sequence.clone()
                changed_sequence[cutoff:] += torch.tensor(
                    (
                        ((3.0, -4.0, 5.0, -6.0), (7.0, -8.0, 9.0, -10.0)),
                        ((-11.0, 12.0, -13.0, 14.0), (15.0, -16.0, 17.0, -18.0)),
                    ),
                    dtype=torch.float64,
                )
                batch_first_flag = layout == "batch_first"
                if layout == "unbatched":
                    base_inputs = base_sequence[:, 0]
                    changed_inputs = changed_sequence[:, 0]
                elif batch_first_flag:
                    base_inputs = base_sequence.transpose(0, 1).contiguous()
                    changed_inputs = changed_sequence.transpose(0, 1).contiguous()
                else:
                    base_inputs = base_sequence
                    changed_inputs = changed_sequence
                base_inputs = base_inputs.clone().requires_grad_()
                changed_inputs = changed_inputs.clone().requires_grad_()
                base_model = self._model(batch_first_flag)
                changed_model = self._model(batch_first_flag)
                changed_model.load_state_dict(base_model.state_dict(), strict=True)

                base_output, _, _ = base_model(
                    base_inputs,
                    base_inputs,
                    base_inputs,
                    attention_mask=explicit_mask,
                )
                changed_output, _, _ = changed_model(
                    changed_inputs,
                    changed_inputs,
                    changed_inputs,
                    attention_mask=explicit_mask,
                )
                if batch_first_flag and layout != "unbatched":
                    base_prefix = base_output[:, :cutoff]
                    changed_prefix = changed_output[:, :cutoff]
                else:
                    base_prefix = base_output[:cutoff]
                    changed_prefix = changed_output[:cutoff]

                torch.testing.assert_close(
                    changed_prefix,
                    base_prefix,
                    rtol=1e-12,
                    atol=1e-12,
                )
                marker = torch.arange(
                    1,
                    base_prefix.numel() + 1,
                    dtype=base_prefix.dtype,
                ).reshape_as(base_prefix)
                (base_prefix * marker).sum().backward()
                (changed_prefix * marker).sum().backward()

                torch.testing.assert_close(
                    changed_inputs.grad,
                    base_inputs.grad,
                    rtol=1e-11,
                    atol=1e-11,
                )
                if batch_first_flag and layout != "unbatched":
                    future_gradient = changed_inputs.grad[:, cutoff:]
                else:
                    future_gradient = changed_inputs.grad[cutoff:]
                torch.testing.assert_close(
                    future_gradient,
                    torch.zeros_like(future_gradient),
                    rtol=0.0,
                    atol=0.0,
                )
                for (base_name, base_parameter), (
                    changed_name,
                    changed_parameter,
                ) in zip(
                    base_model.named_parameters(),
                    changed_model.named_parameters(),
                    strict=True,
                ):
                    with self.subTest(layout=layout, parameter=base_name):
                        self.assertEqual(base_name, changed_name)
                        torch.testing.assert_close(
                            changed_parameter.grad,
                            base_parameter.grad,
                            rtol=1e-11,
                            atol=1e-11,
                        )


class TestRelativePositionAttentionOracle(unittest.TestCase):
    def test_self_and_mixture_match_unscaled_relative_equation_and_gradients(
        self,
    ) -> None:
        for config_class in (
            SelfAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        ):
            with self.subTest(config_class=config_class):
                mixture_options = {}
                if config_class is MixtureOfAttentionHeadsConfig:
                    mixture_options = {
                        "use_kv_expert_models_flag": False,
                        "experts_top_k": 1,
                        "experts_num_experts": 2,
                        "experts_stack_num_layers": 1,
                    }
                config = build_attention_config(
                    config_class=config_class,
                    batch_size=1,
                    num_heads=2,
                    embedding_dim=4,
                    target_sequence_length=3,
                    source_sequence_length=3,
                    return_attention_weights_flag=(config_class is SelfAttentionConfig),
                    self_attention_projection_strategy=(
                        SelfAttentionProjectionStrategy.SEPARATE
                    ),
                    relative_positional_embedding_config_cls=(
                        DynamicPositionalBiasConfig
                    ),
                    **mixture_options,
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

                    relative = model.processor.relative_positional_embedding
                    relative.relative_positional_embeddings.zero_()
                    center = relative.max_positions
                    relative.relative_positional_embeddings[
                        :, :, center - 2 : center + 3
                    ].copy_(
                        torch.tensor(
                            [
                                [
                                    [-0.4, 0.2, 0.7, -0.1, 0.5],
                                    [0.3, -0.6, 0.1, 0.8, -0.2],
                                ],
                                [
                                    [0.9, -0.3, 0.4, -0.7, 0.2],
                                    [-0.5, 0.6, -0.8, 0.3, 0.1],
                                ],
                            ],
                            dtype=torch.float64,
                        )
                    )

                inputs = torch.tensor(
                    [
                        [[1.0, 0.5, -0.5, 2.0]],
                        [[-0.75, 1.25, 1.0, -1.0]],
                        [[2.0, -0.25, 0.5, 0.75]],
                    ],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                reference_inputs = inputs.detach().clone().requires_grad_()
                reference_table = (
                    relative.relative_positional_embeddings.detach()
                    .clone()
                    .requires_grad_()
                )

                output, weights, auxiliary_loss = model(inputs, inputs, inputs)
                reference_output, reference_weights = (
                    _reference_attention_with_relative_bias(
                        reference_inputs,
                        reference_table,
                        num_heads=2,
                        max_positions=relative.max_positions,
                    )
                )
                if config_class is MixtureOfAttentionHeadsConfig:
                    reference_output = reference_output * 0.5

                torch.testing.assert_close(
                    output,
                    reference_output,
                    rtol=1e-12,
                    atol=1e-12,
                )
                self.assertTrue(torch.any(output.abs() > 0))
                if config_class is SelfAttentionConfig:
                    torch.testing.assert_close(
                        weights,
                        reference_weights,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    self.assertIsNone(auxiliary_loss)
                else:
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
                objective = (output * marker).sum()
                if auxiliary_loss is not None:
                    objective = objective + auxiliary_loss
                objective.backward()
                (reference_output * marker).sum().backward()

                torch.testing.assert_close(
                    inputs.grad,
                    reference_inputs.grad,
                    rtol=1e-11,
                    atol=1e-11,
                )
                torch.testing.assert_close(
                    relative.relative_positional_embeddings.grad,
                    reference_table.grad,
                    rtol=1e-11,
                    atol=1e-11,
                )
                self.assertTrue(torch.any(inputs.grad.abs() > 0))
                self.assertTrue(torch.any(reference_table.grad.abs() > 0))


class TestAsymmetricMixtureAttentionOracle(unittest.TestCase):
    def test_attention_dimensions_override_unrelated_nested_expert_dimensions(
        self,
    ) -> None:
        config = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=6,
            query_key_projection_dim=4,
            value_projection_dim=2,
            target_sequence_length=3,
            source_sequence_length=3,
            use_kv_expert_models_flag=True,
            experts_top_k=1,
            experts_num_experts=3,
            experts_stack_num_layers=1,
        )
        experts_config = config.experts_config
        sampler_config = experts_config.sampler_config
        poisoned_config = replace(
            config,
            experts_config=replace(
                experts_config,
                input_dim=11,
                output_dim=13,
                sampler_config=replace(
                    sampler_config,
                    router_config=replace(
                        sampler_config.router_config,
                        input_dim=17,
                    ),
                ),
            ),
        )

        torch.manual_seed(923)
        canonical_model = replace(
            config,
            target_dtype=torch.float64,
            batch_first_flag=False,
        ).build()
        canonical_model.eval()
        torch.manual_seed(923)
        poisoned_model = replace(
            poisoned_config,
            target_dtype=torch.float64,
            batch_first_flag=False,
        ).build()
        poisoned_model.eval()

        canonical_inputs = (
            torch.arange(36, dtype=torch.float64)
            .reshape(3, 2, 6)
            .div_(11.0)
            .sub_(1.0)
            .requires_grad_()
        )
        poisoned_inputs = canonical_inputs.detach().clone().requires_grad_()

        canonical_output, canonical_weights, canonical_auxiliary = canonical_model(
            canonical_inputs,
            canonical_inputs,
            canonical_inputs,
        )
        poisoned_output, poisoned_weights, poisoned_auxiliary = poisoned_model(
            poisoned_inputs,
            poisoned_inputs,
            poisoned_inputs,
        )

        torch.testing.assert_close(
            poisoned_output,
            canonical_output,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            poisoned_auxiliary,
            canonical_auxiliary,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertIsNone(canonical_weights)
        self.assertIsNone(poisoned_weights)
        self.assertTrue(torch.isfinite(poisoned_output).all())
        self.assertTrue(torch.any(poisoned_output.abs() > 0))

        marker = torch.arange(
            1,
            canonical_output.numel() + 1,
            dtype=canonical_output.dtype,
        ).reshape_as(canonical_output)
        ((canonical_output * marker).sum() + canonical_auxiliary).backward()
        ((poisoned_output * marker).sum() + poisoned_auxiliary).backward()

        torch.testing.assert_close(
            poisoned_inputs.grad,
            canonical_inputs.grad,
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertTrue(torch.isfinite(poisoned_inputs.grad).all())
        self.assertTrue(torch.any(poisoned_inputs.grad.abs() > 0))

    def test_distinct_batch_head_expert_and_sequence_axes_match_reference(
        self,
    ) -> None:
        batch_size = 2
        num_heads = 3
        top_k = 4
        num_experts = 5
        target_length = 5
        source_length = 6
        embedding_dim = 6
        config = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=batch_size,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            target_sequence_length=target_length,
            source_sequence_length=source_length,
            use_kv_expert_models_flag=False,
            experts_top_k=top_k,
            experts_num_experts=num_experts,
            experts_stack_num_layers=1,
        )
        model = replace(
            config,
            target_dtype=torch.float64,
            batch_first_flag=False,
        ).build()
        model.eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for expert in model.projector.query_model.expert_modules:
                _set_stack_identity(expert)
            for expert in model.projector.output_model.expert_modules:
                _set_stack_identity(expert)
            _set_stack_identity(model.projector.key_model)
            _set_stack_identity(model.projector.value_model)

        query = (
            torch.arange(
                target_length * batch_size * embedding_dim,
                dtype=torch.float64,
            )
            .reshape(target_length, batch_size, embedding_dim)
            .div_(17.0)
            .sub_(1.5)
            .requires_grad_()
        )
        key = (
            torch.arange(
                source_length * batch_size * embedding_dim,
                dtype=torch.float64,
            )
            .flip(0)
            .reshape(source_length, batch_size, embedding_dim)
            .div_(19.0)
            .sub_(1.0)
            .requires_grad_()
        )
        value = (
            torch.arange(
                source_length * batch_size * embedding_dim,
                dtype=torch.float64,
            )
            .roll(7)
            .reshape(source_length, batch_size, embedding_dim)
            .div_(13.0)
            .sub_(2.0)
            .requires_grad_()
        )
        reference_query = query.detach().clone().requires_grad_()
        reference_key = key.detach().clone().requires_grad_()
        reference_value = value.detach().clone().requires_grad_()

        output, weights, auxiliary_loss = model(query, key, value)
        reference_output, _ = _reference_attention(
            reference_query,
            reference_key,
            reference_value,
            num_heads=num_heads,
        )
        selected_probability_mass = top_k / num_experts
        reference_output = reference_output * selected_probability_mass

        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertIsNone(weights)
        torch.testing.assert_close(auxiliary_loss, auxiliary_loss.new_zeros(()))
        self.assertIsNone(model.projector.probabilities)
        self.assertIsNone(model.projector.indices)
        self.assertIsNone(model.projector.skip_mask)

        marker = torch.arange(
            1,
            output.numel() + 1,
            dtype=output.dtype,
        ).reshape_as(output)
        ((output * marker).sum() + auxiliary_loss).backward()
        (reference_output * marker).sum().backward()
        for actual, expected in (
            (query.grad, reference_query.grad),
            (key.grad, reference_key.grad),
            (value.grad, reference_value.grad),
        ):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-11,
                atol=1e-11,
            )


if __name__ == "__main__":
    unittest.main()
