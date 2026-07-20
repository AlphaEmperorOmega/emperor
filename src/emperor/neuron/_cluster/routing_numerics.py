import math
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


class _ForwardValueWithSurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        forward_value: Tensor,
        surrogate_value: Tensor,
    ) -> Tensor:
        ctx.surrogate_dtype = surrogate_value.dtype
        return forward_value

    @staticmethod
    def backward(ctx: Any, output_gradient: Tensor) -> tuple[None, Tensor]:
        return None, output_gradient.to(dtype=ctx.surrogate_dtype)


def _forward_value_with_surrogate_gradient(
    forward_value: Tensor,
    surrogate_value: Tensor,
) -> Tensor:
    return _ForwardValueWithSurrogateGradient.apply(
        forward_value,
        surrogate_value,
    )


def _log_space_route_weights(
    log_probabilities: Tensor,
    valid_branch_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    finite_valid_mask = valid_branch_mask & torch.isfinite(log_probabilities)
    has_finite_valid_branch = finite_valid_mask.any(dim=1, keepdim=True)
    # Log-probabilities all contain the same global log-softmax normalizer.
    # Centre on one valid branch so that common term cancels explicitly,
    # rather than only algebraically.
    reference_indices = finite_valid_mask.to(torch.int64).argmax(
        dim=1,
        keepdim=True,
    )
    reference_log_probabilities = log_probabilities.gather(
        1,
        reference_indices,
    )
    relative_log_probabilities = log_probabilities - reference_log_probabilities
    masked_log_probabilities = torch.where(
        finite_valid_mask,
        relative_log_probabilities,
        torch.full_like(log_probabilities, float("-inf")),
    )
    safe_masked_log_probabilities = torch.where(
        has_finite_valid_branch,
        masked_log_probabilities,
        torch.zeros_like(masked_log_probabilities),
    )
    conditional_weights = torch.softmax(
        safe_masked_log_probabilities,
        dim=1,
    ) * finite_valid_mask.to(log_probabilities.dtype)
    unconditional_weights = torch.exp(log_probabilities)
    single_finite_valid_branch = finite_valid_mask.sum(dim=1, keepdim=True) == 1
    surrogate_weights = torch.where(
        single_finite_valid_branch,
        unconditional_weights * valid_branch_mask,
        conditional_weights,
    )
    surrogate_weights = torch.where(
        has_finite_valid_branch,
        surrogate_weights,
        unconditional_weights,
    )
    return conditional_weights, surrogate_weights, finite_valid_mask


def _conditional_log_route_weights(
    log_probabilities: Tensor,
    valid_branch_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    finite_valid_mask = valid_branch_mask & torch.isfinite(log_probabilities)
    has_finite_valid_branch = finite_valid_mask.any(dim=1, keepdim=True)
    masked_for_reference = torch.where(
        finite_valid_mask,
        log_probabilities,
        torch.full_like(log_probabilities, float("-inf")),
    )
    reference_log_probabilities = masked_for_reference.max(
        dim=1,
        keepdim=True,
    ).values
    reference_log_probabilities = torch.where(
        has_finite_valid_branch,
        reference_log_probabilities,
        torch.zeros_like(reference_log_probabilities),
    )
    relative_log_probabilities = log_probabilities - reference_log_probabilities
    masked_log_probabilities = torch.where(
        finite_valid_mask,
        relative_log_probabilities,
        torch.full_like(relative_log_probabilities, float("-inf")),
    )
    safe_log_probabilities = torch.where(
        has_finite_valid_branch,
        masked_log_probabilities,
        torch.zeros_like(masked_log_probabilities),
    )
    log_normalizer = torch.logsumexp(
        safe_log_probabilities,
        dim=1,
        keepdim=True,
    )
    conditional_log_weights = torch.where(
        finite_valid_mask,
        safe_log_probabilities - log_normalizer,
        torch.full_like(safe_log_probabilities, float("-inf")),
    )
    return conditional_log_weights, finite_valid_mask


def _scaled_values_from_log_weights(
    values: Tensor,
    log_weights: Tensor,
) -> Tensor:
    computation_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    working_values = values.to(computation_dtype)
    working_log_weights = log_weights.to(computation_dtype)
    ordinary_weights = torch.exp(working_log_weights)
    ordinary_terms = working_values * ordinary_weights.unsqueeze(-1)
    finite_log_weight_mask = torch.isfinite(working_log_weights)
    underflowed_weight_mask = finite_log_weight_mask & (
        ordinary_weights.abs() < torch.finfo(computation_dtype).tiny
    )
    unstable_product_mask = (
        finite_log_weight_mask.unsqueeze(-1)
        & torch.isfinite(working_values)
        & ~torch.isfinite(ordinary_terms)
    )
    log_product_mask = underflowed_weight_mask.unsqueeze(-1) | unstable_product_mask
    if not bool(log_product_mask.any().item()):
        return ordinary_terms

    double_values = values.double()
    nonzero_value_mask = double_values != 0
    safe_magnitudes = torch.where(
        nonzero_value_mask,
        double_values.abs(),
        torch.ones_like(double_values),
    )
    stable_terms = torch.where(
        nonzero_value_mask,
        double_values.sign()
        * torch.exp(log_weights.double().unsqueeze(-1) + torch.log(safe_magnitudes)),
        torch.zeros_like(double_values),
    ).to(computation_dtype)
    return torch.where(
        log_product_mask,
        stable_terms,
        ordinary_terms,
    )


def _signed_factors_times_log_scale(
    first_factor: Tensor,
    second_factor: Tensor,
    log_scale: Tensor,
) -> Tensor:
    double_first_factor = first_factor.double()
    double_second_factor = second_factor.double()
    double_log_scale = log_scale.double()
    ordinary_scale = torch.exp(double_log_scale)
    unscaled_product = double_first_factor * double_second_factor
    ordinary_product = unscaled_product * ordinary_scale
    finite_factor_mask = torch.isfinite(double_first_factor) & torch.isfinite(
        double_second_factor
    )
    nonzero_factor_mask = (double_first_factor != 0) & (double_second_factor != 0)
    log_product_mask = (
        torch.isfinite(double_log_scale)
        & finite_factor_mask
        & nonzero_factor_mask
        & (
            (ordinary_scale.abs() < torch.finfo(torch.float64).tiny)
            | ~torch.isfinite(unscaled_product)
            | ~torch.isfinite(ordinary_product)
        )
    )
    safe_first_magnitude = torch.where(
        double_first_factor != 0,
        double_first_factor.abs(),
        torch.ones_like(double_first_factor),
    )
    safe_second_magnitude = torch.where(
        double_second_factor != 0,
        double_second_factor.abs(),
        torch.ones_like(double_second_factor),
    )
    log_product = (
        double_first_factor.sign()
        * double_second_factor.sign()
        * torch.exp(
            double_log_scale
            + torch.log(safe_first_magnitude)
            + torch.log(safe_second_magnitude)
        )
    )
    return torch.where(log_product_mask, log_product, ordinary_product)


def _compensated_add(
    total: Tensor,
    compensation: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor]:
    next_total = total + value
    finite_correction = torch.where(
        total.abs() >= value.abs(),
        (total - next_total) + value,
        (value - next_total) + total,
    )
    correction = torch.where(
        torch.isfinite(total) & torch.isfinite(value) & torch.isfinite(next_total),
        finite_correction,
        torch.zeros_like(finite_correction),
    )
    return next_total, compensation + correction


def _compensated_sum_expansion_last_dimension(
    values: Tensor,
) -> tuple[Tensor, Tensor]:
    total = values.new_zeros(values.shape[:-1])
    compensation = torch.zeros_like(total)
    for value_index in range(values.shape[-1]):
        total, compensation = _compensated_add(
            total,
            compensation,
            values[..., value_index],
        )
    return total, compensation


def _compensated_sum_last_dimension(values: Tensor) -> Tensor:
    total, compensation = _compensated_sum_expansion_last_dimension(values)
    return total + compensation


def _sum_signed_products_in_log_space(
    first_factors: Tensor,
    second_factors: Tensor,
    log_scale: Tensor,
    direct_scale: Tensor | None = None,
) -> Tensor:
    first_factors = first_factors.double()
    second_factors = second_factors.double()
    log_scale = log_scale.double()
    term_log_scales = (
        log_scale
        if log_scale.shape == first_factors.shape
        else log_scale.unsqueeze(-1).expand_as(first_factors)
    )
    finite_nonzero_factor_mask = (
        torch.isfinite(first_factors)
        & torch.isfinite(second_factors)
        & (first_factors != 0)
        & (second_factors != 0)
        & torch.isfinite(term_log_scales)
    )
    safe_first_magnitudes = torch.where(
        finite_nonzero_factor_mask,
        first_factors.abs(),
        torch.ones_like(first_factors),
    )
    safe_second_magnitudes = torch.where(
        finite_nonzero_factor_mask,
        second_factors.abs(),
        torch.ones_like(second_factors),
    )
    first_mantissas, first_exponents = torch.frexp(safe_first_magnitudes)
    second_mantissas, second_exponents = torch.frexp(safe_second_magnitudes)
    mantissa_products = first_mantissas * second_mantissas
    splitter = mantissa_products.new_tensor(134217729.0)
    split_first = splitter * first_mantissas
    first_high = split_first - (split_first - first_mantissas)
    first_low = first_mantissas - first_high
    split_second = splitter * second_mantissas
    second_high = split_second - (split_second - second_mantissas)
    second_low = second_mantissas - second_high
    product_roundoff = (
        (first_high * second_high - mantissa_products)
        + first_high * second_low
        + first_low * second_high
        + first_low * second_low
    )
    normalized_product_high, normalization_exponents = torch.frexp(mantissa_products)
    normalized_product_low = torch.ldexp(
        product_roundoff,
        -normalization_exponents,
    )
    product_exponents = first_exponents + second_exponents + normalization_exponents
    normalized_product_magnitudes = normalized_product_high + normalized_product_low
    product_log_magnitudes = (
        torch.log(normalized_product_magnitudes)
        + product_exponents.double() * math.log(2.0)
        + term_log_scales
    )
    product_signs = first_factors.sign() * second_factors.sign()

    ungrouped_product_mask = finite_nonzero_factor_mask.clone()
    grouped_coefficients = torch.zeros_like(product_log_magnitudes)
    grouped_log_magnitudes = torch.full_like(
        product_log_magnitudes,
        float("-inf"),
    )
    grouped_product_magnitudes = torch.zeros_like(product_log_magnitudes)
    grouped_product_exponents = torch.zeros_like(product_exponents)
    # Equal-and-opposite products must cancel before exponentiation. Otherwise
    # two individually overflowing feature products can become inf - inf even
    # when their dot-product residual is representable.
    for feature_index in range(first_factors.shape[-1]):
        representative_is_ungrouped = ungrouped_product_mask[..., feature_index]
        representative_log_magnitude = product_log_magnitudes[..., feature_index]
        representative_product_exponent = product_exponents[..., feature_index]
        representative_product_high = normalized_product_high[..., feature_index]
        representative_product_low = normalized_product_low[..., feature_index]
        representative_log_scale = term_log_scales[..., feature_index]
        equal_magnitude_group_mask = (
            ungrouped_product_mask
            & representative_is_ungrouped.unsqueeze(-1)
            & (term_log_scales == representative_log_scale.unsqueeze(-1))
            & (product_exponents == representative_product_exponent.unsqueeze(-1))
            & (normalized_product_high == representative_product_high.unsqueeze(-1))
            & (normalized_product_low == representative_product_low.unsqueeze(-1))
        )
        grouped_coefficients[..., feature_index] = torch.where(
            representative_is_ungrouped,
            torch.where(
                equal_magnitude_group_mask,
                product_signs,
                torch.zeros_like(product_signs),
            ).sum(dim=-1),
            torch.zeros_like(representative_log_magnitude),
        )
        grouped_log_magnitudes[..., feature_index] = torch.where(
            representative_is_ungrouped,
            representative_log_magnitude,
            torch.full_like(representative_log_magnitude, float("-inf")),
        )
        grouped_product_magnitudes[..., feature_index] = torch.where(
            representative_is_ungrouped,
            normalized_product_magnitudes[..., feature_index],
            torch.zeros_like(representative_log_magnitude),
        )
        grouped_product_exponents[..., feature_index] = torch.where(
            representative_is_ungrouped,
            representative_product_exponent,
            torch.zeros_like(representative_product_exponent),
        )
        ungrouped_product_mask &= ~equal_magnitude_group_mask

    nonzero_group_mask = grouped_coefficients != 0
    direct_group_sum = None
    direct_group_sum_is_usable = None
    if direct_scale is not None:
        double_direct_scale = direct_scale.double()
        term_direct_scales = (
            double_direct_scale
            if double_direct_scale.shape == first_factors.shape
            else double_direct_scale.unsqueeze(-1).expand_as(first_factors)
        )
        scale_mantissas, scale_exponents = torch.frexp(term_direct_scales)
        scaled_group_mantissas = (
            grouped_coefficients * grouped_product_magnitudes * scale_mantissas
        )
        normalized_scaled_mantissas, scaled_normalization_exponents = torch.frexp(
            scaled_group_mantissas
        )
        direct_group_terms = torch.ldexp(
            normalized_scaled_mantissas,
            grouped_product_exponents
            + scale_exponents
            + scaled_normalization_exponents,
        )
        direct_group_terms = torch.where(
            nonzero_group_mask,
            direct_group_terms,
            torch.zeros_like(direct_group_terms),
        )
        direct_group_sum = _compensated_sum_last_dimension(direct_group_terms)
        direct_group_sum_is_usable = (
            ~nonzero_group_mask
            | (
                (
                    (term_direct_scales.abs() >= torch.finfo(torch.float64).tiny)
                    | ~torch.isfinite(term_log_scales)
                )
                & torch.isfinite(direct_group_terms)
            )
        ).all(dim=-1) & torch.isfinite(direct_group_sum)
    reference_log_magnitude = torch.where(
        nonzero_group_mask,
        grouped_log_magnitudes,
        torch.full_like(grouped_log_magnitudes, float("-inf")),
    ).amax(dim=-1)
    has_nonzero_group = nonzero_group_mask.any(dim=-1)
    safe_reference_log_magnitude = torch.where(
        has_nonzero_group,
        reference_log_magnitude,
        torch.zeros_like(reference_log_magnitude),
    )
    scaled_group_values = torch.where(
        nonzero_group_mask,
        grouped_coefficients
        * torch.exp(
            grouped_log_magnitudes - safe_reference_log_magnitude.unsqueeze(-1)
        ),
        torch.zeros_like(grouped_coefficients),
    )
    scaled_sum = _compensated_sum_last_dimension(scaled_group_values)
    nonzero_sum_mask = scaled_sum != 0
    safe_scaled_sum = torch.where(
        nonzero_sum_mask,
        scaled_sum.abs(),
        torch.ones_like(scaled_sum),
    )
    log_space_sum = torch.where(
        has_nonzero_group & nonzero_sum_mask,
        scaled_sum.sign()
        * torch.exp(torch.log(safe_scaled_sum) + safe_reference_log_magnitude),
        torch.zeros_like(scaled_sum),
    )
    if direct_group_sum is None or direct_group_sum_is_usable is None:
        return log_space_sum
    return torch.where(
        direct_group_sum_is_usable,
        direct_group_sum,
        log_space_sum,
    )


def _dot_product_times_log_scale(
    first_factors: Tensor,
    second_factors: Tensor,
    log_scale: Tensor,
    direct_scale: Tensor | None = None,
) -> Tensor:
    double_first_factors = first_factors.double()
    double_second_factors = second_factors.double()
    ordinary_products = double_first_factors * double_second_factors
    ordinary_dot_product = ordinary_products.sum(dim=-1)
    ordinary_projection_is_finite = torch.isfinite(ordinary_products).all(
        dim=-1
    ) & torch.isfinite(ordinary_dot_product)
    log_scaled_projection = _signed_factors_times_log_scale(
        ordinary_dot_product,
        torch.ones_like(ordinary_dot_product),
        log_scale,
    )
    if direct_scale is None:
        directly_scaled_projection = log_scaled_projection
        direct_projection_is_usable = ordinary_projection_is_finite
    else:
        double_direct_scale = direct_scale.double()
        directly_scaled_projection = ordinary_dot_product * double_direct_scale
        direct_projection_is_usable = (
            ordinary_projection_is_finite
            & torch.isfinite(directly_scaled_projection)
            & (
                (double_direct_scale.abs() >= torch.finfo(torch.float64).tiny)
                | ~torch.isfinite(log_scale)
                | (ordinary_dot_product == 0)
            )
        )
        directly_scaled_projection = torch.where(
            direct_projection_is_usable,
            directly_scaled_projection,
            log_scaled_projection,
        )
    if bool(direct_projection_is_usable.all().item()):
        return directly_scaled_projection
    log_space_projection = _sum_signed_products_in_log_space(
        double_first_factors,
        double_second_factors,
        log_scale,
        direct_scale,
    )
    return torch.where(
        direct_projection_is_usable,
        directly_scaled_projection,
        log_space_projection,
    )


def _stable_weighted_average(values: Tensor, log_weights: Tensor) -> Tensor:
    weighted_sum = _sum_values_times_log_scales(values, log_weights)
    partition_mass = torch.exp(log_weights).sum(dim=1, keepdim=True)
    safe_partition_mass = torch.where(
        partition_mass > 0,
        partition_mass,
        torch.ones_like(partition_mass),
    )
    return weighted_sum / safe_partition_mass


def _sum_values_times_log_scales(
    values: Tensor,
    log_scales: Tensor,
) -> Tensor:
    computation_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    working_values = values.to(computation_dtype)
    finite_scale_mask = torch.isfinite(log_scales)
    ungrouped_scale_mask = finite_scale_mask.clone()
    weighted_sum = working_values.new_zeros((values.shape[0], values.shape[2]))
    weighted_compensation = torch.zeros_like(weighted_sum)
    for route_index in range(values.shape[1]):
        representative_is_ungrouped = ungrouped_scale_mask[:, route_index]
        equal_scale_group_mask = (
            ungrouped_scale_mask
            & representative_is_ungrouped.unsqueeze(1)
            & (log_scales == log_scales[:, route_index].unsqueeze(1))
        )
        grouped_values = torch.where(
            equal_scale_group_mask.unsqueeze(-1),
            working_values,
            torch.zeros_like(working_values),
        )
        group_size = equal_scale_group_mask.sum(dim=1, keepdim=True).clamp_min(1)
        group_maximum_magnitude = grouped_values.abs().amax(dim=1)
        overflow_risk_mask = group_maximum_magnitude > (
            torch.finfo(computation_dtype).max / group_size.to(computation_dtype)
        )
        # Dividing an overflow-prone group by the next power of two at least
        # as large as its cardinality guarantees headroom for the reduction.
        # Keep ordinary groups unscaled so representable subnormals survive.
        group_divisor = torch.exp2(
            torch.ceil(torch.log2(group_size.to(computation_dtype)))
        )
        reduction_divisor = torch.where(
            overflow_risk_mask,
            group_divisor,
            torch.ones_like(group_divisor),
        )
        scaled_group_values = grouped_values / reduction_divisor.unsqueeze(1)
        group_sum = working_values.new_zeros((values.shape[0], values.shape[2]))
        group_compensation = torch.zeros_like(group_sum)
        for grouped_route_index in range(values.shape[1]):
            route_values = scaled_group_values[:, grouped_route_index]
            next_group_sum = group_sum + route_values
            correction = torch.where(
                group_sum.abs() >= route_values.abs(),
                (group_sum - next_group_sum) + route_values,
                (route_values - next_group_sum) + group_sum,
            )
            group_compensation = group_compensation + correction
            group_sum = next_group_sum
        group_log_scale = torch.where(
            representative_is_ungrouped,
            log_scales[:, route_index],
            torch.full_like(log_scales[:, route_index], float("-inf")),
        )
        adjusted_group_log_scale = group_log_scale.unsqueeze(-1) + torch.log(
            reduction_divisor
        )
        for unscaled_group_contribution in (group_sum, group_compensation):
            group_contribution = _signed_factors_times_log_scale(
                unscaled_group_contribution,
                torch.ones_like(unscaled_group_contribution),
                adjusted_group_log_scale,
            ).to(computation_dtype)
            next_weighted_sum = weighted_sum + group_contribution
            weighted_correction = torch.where(
                weighted_sum.abs() >= group_contribution.abs(),
                (weighted_sum - next_weighted_sum) + group_contribution,
                (group_contribution - next_weighted_sum) + weighted_sum,
            )
            weighted_compensation = weighted_compensation + weighted_correction
            weighted_sum = next_weighted_sum
        ungrouped_scale_mask = ungrouped_scale_mask & ~equal_scale_group_mask
    return weighted_sum + weighted_compensation


def _difference_with_roundoff(
    left_values: Tensor,
    right_values: Tensor,
) -> tuple[Tensor, Tensor]:
    differences = left_values - right_values
    finite_differences = (
        torch.isfinite(left_values)
        & torch.isfinite(right_values)
        & torch.isfinite(differences)
    )
    virtual_right_values = left_values - differences
    virtual_left_values = differences + virtual_right_values
    right_roundoff = virtual_right_values - right_values
    left_roundoff = left_values - virtual_left_values
    difference_roundoff = left_roundoff + right_roundoff
    return differences, torch.where(
        finite_differences,
        difference_roundoff,
        torch.zeros_like(difference_roundoff),
    )


def _sum_projections_times_scales(
    projections: Tensor,
    projection_roundoff: Tensor,
    scales: Tensor,
) -> Tensor:
    ungrouped_scale_mask = torch.ones_like(scales, dtype=torch.bool)
    weighted_sum = projections.new_zeros(projections.shape[0])
    weighted_compensation = torch.zeros_like(weighted_sum)
    for route_index in range(projections.shape[1]):
        representative_is_ungrouped = ungrouped_scale_mask[:, route_index]
        equal_scale_group_mask = (
            ungrouped_scale_mask
            & representative_is_ungrouped.unsqueeze(1)
            & (scales == scales[:, route_index].unsqueeze(1))
        )
        group_sum = projections.new_zeros(projections.shape[0])
        group_compensation = torch.zeros_like(group_sum)
        for group_values in (projections, projection_roundoff):
            for grouped_route_index in range(projections.shape[1]):
                route_values = torch.where(
                    equal_scale_group_mask[:, grouped_route_index],
                    group_values[:, grouped_route_index],
                    torch.zeros_like(group_values[:, grouped_route_index]),
                )
                next_group_sum = group_sum + route_values
                correction = torch.where(
                    group_sum.abs() >= route_values.abs(),
                    (group_sum - next_group_sum) + route_values,
                    (route_values - next_group_sum) + group_sum,
                )
                group_compensation = group_compensation + correction
                group_sum = next_group_sum

        group_scale = torch.where(
            representative_is_ungrouped,
            scales[:, route_index],
            torch.zeros_like(scales[:, route_index]),
        )
        for unscaled_group_contribution in (group_sum, group_compensation):
            group_contribution = unscaled_group_contribution * group_scale
            next_weighted_sum = weighted_sum + group_contribution
            correction = torch.where(
                weighted_sum.abs() >= group_contribution.abs(),
                (weighted_sum - next_weighted_sum) + group_contribution,
                (group_contribution - next_weighted_sum) + weighted_sum,
            )
            weighted_compensation = weighted_compensation + correction
            weighted_sum = next_weighted_sum
        ungrouped_scale_mask = ungrouped_scale_mask & ~equal_scale_group_mask
    return weighted_sum + weighted_compensation


def _sum_feature_products_times_scales(
    products: Tensor,
    product_roundoff: Tensor,
    scales: Tensor,
) -> Tensor:
    ungrouped_scale_mask = torch.ones_like(scales, dtype=torch.bool)
    weighted_sum = products.new_zeros(products.shape[0])
    weighted_compensation = torch.zeros_like(weighted_sum)
    for route_index in range(products.shape[1]):
        representative_is_ungrouped = ungrouped_scale_mask[:, route_index]
        equal_scale_group_mask = (
            ungrouped_scale_mask
            & representative_is_ungrouped.unsqueeze(1)
            & (scales == scales[:, route_index].unsqueeze(1))
        )
        group_sum = products.new_zeros(products.shape[0])
        group_compensation = torch.zeros_like(group_sum)
        for group_values in (products, product_roundoff):
            for grouped_route_index in range(products.shape[1]):
                for feature_index in range(products.shape[2]):
                    feature_product = torch.where(
                        equal_scale_group_mask[:, grouped_route_index],
                        group_values[:, grouped_route_index, feature_index],
                        torch.zeros_like(
                            group_values[:, grouped_route_index, feature_index]
                        ),
                    )
                    next_group_sum = group_sum + feature_product
                    correction = torch.where(
                        group_sum.abs() >= feature_product.abs(),
                        (group_sum - next_group_sum) + feature_product,
                        (feature_product - next_group_sum) + group_sum,
                    )
                    group_compensation = group_compensation + correction
                    group_sum = next_group_sum

        group_scale = torch.where(
            representative_is_ungrouped,
            scales[:, route_index],
            torch.zeros_like(scales[:, route_index]),
        )
        for unscaled_group_contribution in (group_sum, group_compensation):
            group_contribution = unscaled_group_contribution * group_scale
            next_weighted_sum = weighted_sum + group_contribution
            correction = torch.where(
                weighted_sum.abs() >= group_contribution.abs(),
                (weighted_sum - next_weighted_sum) + group_contribution,
                (group_contribution - next_weighted_sum) + weighted_sum,
            )
            weighted_compensation = weighted_compensation + correction
            weighted_sum = next_weighted_sum
        ungrouped_scale_mask = ungrouped_scale_mask & ~equal_scale_group_mask
    return weighted_sum + weighted_compensation


def _row_has_finite_difference_overflow(
    values: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    detached_values = values.detach()
    expanded_valid_mask = valid_mask.unsqueeze(-1)
    maximum_values = torch.where(
        expanded_valid_mask,
        detached_values,
        torch.full_like(detached_values, float("-inf")),
    ).amax(dim=1)
    minimum_values = torch.where(
        expanded_valid_mask,
        detached_values,
        torch.full_like(detached_values, float("inf")),
    ).amin(dim=1)
    finite_extrema = torch.isfinite(maximum_values) & torch.isfinite(minimum_values)
    overflowing_range_by_feature = finite_extrema & ~torch.isfinite(
        maximum_values - minimum_values
    )
    return overflowing_range_by_feature.any(dim=-1, keepdim=True)


def _row_has_equal_weight_cancellation(
    values: Tensor,
    log_weights: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    detached_values = values.detach()
    cancellation_risk_mask = torch.zeros(
        values.shape[0],
        dtype=torch.bool,
        device=values.device,
    )
    for left_index in range(values.shape[1]):
        for right_index in range(left_index + 1, values.shape[1]):
            equal_valid_weight_mask = (
                valid_mask[:, left_index]
                & valid_mask[:, right_index]
                & (log_weights[:, left_index] == log_weights[:, right_index])
            )
            opposite_nonzero_sign_mask = (
                (detached_values[:, left_index] > 0)
                & (detached_values[:, right_index] < 0)
            ) | (
                (detached_values[:, left_index] < 0)
                & (detached_values[:, right_index] > 0)
            )
            cancellation_risk_mask |= (
                equal_valid_weight_mask & opposite_nonzero_sign_mask.any(dim=-1)
            )
    return cancellation_risk_mask.unsqueeze(1)


def _row_has_opposing_centered_values(
    values: Tensor,
    center: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    centered_values = values.detach() - center.detach().unsqueeze(1)
    expanded_valid_mask = valid_mask.unsqueeze(-1)
    has_positive_values = ((centered_values > 0) & expanded_valid_mask).any(dim=1)
    has_negative_values = ((centered_values < 0) & expanded_valid_mask).any(dim=1)
    return (has_positive_values & has_negative_values).any(
        dim=-1,
        keepdim=True,
    )


def _row_has_equal_valid_scores(
    scores: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    reference_indices = valid_mask.to(torch.int64).argmax(dim=1, keepdim=True)
    reference_scores = scores.gather(1, reference_indices)
    return (~valid_mask | (scores == reference_scores)).all(dim=1, keepdim=True) & (
        valid_mask.sum(dim=1, keepdim=True) >= 2
    )


class _StableWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        values: Tensor,
        log_scores: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        working_log_scores = (
            log_scores.float()
            if log_scores.dtype in (torch.float16, torch.bfloat16)
            else log_scores
        )
        conditional_log_weights, finite_valid_mask = _conditional_log_route_weights(
            working_log_scores, valid_mask
        )
        conditional_weights = torch.exp(conditional_log_weights)
        ctx.save_for_backward(
            values,
            conditional_log_weights,
            conditional_weights,
            finite_valid_mask,
        )
        ctx.log_scores_dtype = log_scores.dtype
        return _stable_weighted_average(
            values,
            conditional_log_weights,
        ).to(values.dtype)

    @staticmethod
    def backward(
        ctx: Any,
        output_gradient: Tensor,
    ) -> tuple[Tensor, Tensor, None]:
        values, log_weights, conditional_weights, finite_valid_mask = ctx.saved_tensors
        expanded_output_gradient = output_gradient.unsqueeze(1).expand_as(values)
        value_gradients = _scaled_values_from_log_weights(
            expanded_output_gradient,
            log_weights,
        ).to(values.dtype)

        double_values = values.double()
        double_output_gradient = output_gradient.double()
        stable_mean = _stable_weighted_average(values, log_weights).double()
        score_gradients = log_weights.new_zeros(
            log_weights.shape,
            dtype=torch.float64,
        )
        for route_index in range(values.shape[1]):
            centered_values = double_values[:, route_index] - stable_mean
            centered_products = centered_values * double_output_gradient
            centered_projection = centered_products.sum(dim=-1)
            centered_projection_is_finite = (
                torch.isfinite(centered_values).all(dim=-1)
                & torch.isfinite(centered_products).all(dim=-1)
                & torch.isfinite(centered_projection)
            )
            direct_score_gradient = _dot_product_times_log_scale(
                centered_values,
                double_output_gradient,
                log_weights[:, route_index],
            )
            left_values = (
                double_values[:, route_index].unsqueeze(1).expand_as(double_values)
            )
            value_differences, difference_roundoff = _difference_with_roundoff(
                left_values, double_values
            )
            pair_log_scales = log_weights + log_weights[:, route_index].unsqueeze(1)
            expanded_output_gradient = double_output_gradient.unsqueeze(1).expand_as(
                double_values
            )
            ordinary_products = value_differences * expanded_output_gradient
            ordinary_dot_products = ordinary_products.sum(dim=-1)
            roundoff_products = difference_roundoff * expanded_output_gradient
            roundoff_dot_products = roundoff_products.sum(dim=-1)
            finite_pair_scale_mask = torch.isfinite(pair_log_scales)
            pair_projection_is_finite = torch.isfinite(ordinary_products).all(
                dim=-1
            ) & torch.isfinite(ordinary_dot_products)
            all_pair_projections_are_finite = (
                ~finite_pair_scale_mask | pair_projection_is_finite
            ).all(dim=1)
            grouped_pair_sum = _sum_values_times_log_scales(
                ordinary_dot_products.unsqueeze(-1),
                pair_log_scales,
            ).squeeze(-1)
            direct_terms = _dot_product_times_log_scale(
                value_differences,
                expanded_output_gradient,
                pair_log_scales,
            )
            overflowed_difference_mask = (
                ~torch.isfinite(value_differences)
                & torch.isfinite(left_values)
                & torch.isfinite(double_values)
            ).any(dim=-1)
            if bool(overflowed_difference_mask.any().item()):
                left_terms = _dot_product_times_log_scale(
                    left_values,
                    expanded_output_gradient,
                    pair_log_scales,
                )
                right_terms = _dot_product_times_log_scale(
                    double_values,
                    expanded_output_gradient,
                    pair_log_scales,
                )
                direct_terms = torch.where(
                    overflowed_difference_mask,
                    left_terms - right_terms,
                    direct_terms,
                )
            pair_sum = torch.where(
                all_pair_projections_are_finite,
                grouped_pair_sum,
                direct_terms.sum(dim=1),
            )
            direct_pair_scales = conditional_weights.double() * conditional_weights[
                :, route_index
            ].double().unsqueeze(1)
            ordinary_pair_sum = _sum_projections_times_scales(
                ordinary_dot_products,
                roundoff_dot_products,
                direct_pair_scales,
            )
            projection_precision = (
                torch.finfo(torch.float32).eps
                if values.dtype in (torch.float16, torch.bfloat16)
                else torch.finfo(values.dtype).eps
            )
            absolute_scaled_pair_product_sum = (
                (ordinary_products.abs() + roundoff_products.abs())
                * direct_pair_scales.unsqueeze(-1)
            ).sum(dim=(1, 2))
            ordinary_pair_sum_is_well_conditioned = (
                absolute_scaled_pair_product_sum == 0
            ) | (
                ordinary_pair_sum.abs()
                > projection_precision * absolute_scaled_pair_product_sum
            )
            if bool((~ordinary_pair_sum_is_well_conditioned).any().item()):
                feature_preserving_pair_sum = _sum_feature_products_times_scales(
                    ordinary_products,
                    roundoff_products,
                    direct_pair_scales,
                )
                ordinary_pair_sum = torch.where(
                    ordinary_pair_sum_is_well_conditioned,
                    ordinary_pair_sum,
                    feature_preserving_pair_sum,
                )
            direct_pair_scale_underflow = (
                (direct_pair_scales == 0) & finite_pair_scale_mask
            ).any(dim=1)
            ordinary_pair_is_finite = (
                torch.isfinite(ordinary_products).all(dim=(1, 2))
                & torch.isfinite(roundoff_products).all(dim=(1, 2))
                & torch.isfinite(ordinary_pair_sum)
                & ~direct_pair_scale_underflow
            )
            absolute_centered_product_sum = centered_products.abs().sum(dim=-1)
            centered_projection_is_well_conditioned = (
                absolute_centered_product_sum == 0
            ) | (
                centered_projection.abs()
                > projection_precision * absolute_centered_product_sum
            )
            fallback_score_gradient = torch.where(
                centered_projection_is_finite
                & (centered_projection_is_well_conditioned | ~torch.isfinite(pair_sum)),
                direct_score_gradient,
                pair_sum,
            )
            score_gradients[:, route_index] = torch.where(
                ordinary_pair_is_finite,
                ordinary_pair_sum,
                fallback_score_gradient,
            )

        nonzero_value_mask = double_values != 0
        nonzero_output_gradient_mask = double_output_gradient != 0
        safe_value_magnitudes = torch.where(
            nonzero_value_mask,
            double_values.abs(),
            torch.ones_like(double_values),
        )
        safe_gradient_magnitudes = torch.where(
            nonzero_output_gradient_mask,
            double_output_gradient.abs(),
            torch.ones_like(double_output_gradient),
        )
        contribution_log_magnitudes = (
            torch.where(
                nonzero_value_mask & nonzero_output_gradient_mask.unsqueeze(1),
                torch.log(safe_value_magnitudes)
                + torch.log(safe_gradient_magnitudes).unsqueeze(1),
                torch.full_like(double_values, float("-inf")),
            ).amax(dim=-1)
            + log_weights.double()
        )
        contribution_log_magnitudes = torch.where(
            finite_valid_mask,
            contribution_log_magnitudes,
            torch.full_like(contribution_log_magnitudes, float("-inf")),
        )
        reference_indices = contribution_log_magnitudes.argmax(
            dim=1,
            keepdim=True,
        )
        reference_mask = (
            torch.zeros_like(finite_valid_mask).scatter_(
                1,
                reference_indices,
                True,
            )
            & finite_valid_mask
        )
        score_gradients = torch.where(
            finite_valid_mask,
            score_gradients,
            torch.zeros_like(score_gradients),
        ).to(ctx.log_scores_dtype)
        nonreference_sum = torch.where(
            finite_valid_mask & ~reference_mask,
            score_gradients,
            torch.zeros_like(score_gradients),
        ).sum(dim=1, keepdim=True)
        score_gradients = torch.where(
            reference_mask,
            -nonreference_sum,
            score_gradients,
        )
        return (
            value_gradients,
            score_gradients,
            None,
        )


def _stable_weighted_sum(
    values: Tensor,
    log_scores: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    return _StableWeightedSum.apply(values, log_scores, valid_mask)


def _values_times_partition_mass(
    values: Tensor,
    log_mass: Tensor,
    direct_mass: Tensor,
) -> Tensor:
    double_values = values.double()
    double_direct_mass = direct_mass.double()
    direct_values = double_values * double_direct_mass.unsqueeze(-1)
    log_scaled_values = _scaled_values_from_log_weights(double_values, log_mass)
    direct_values_are_usable = torch.isfinite(direct_values) & (
        (double_direct_mass.abs() >= torch.finfo(torch.float64).tiny)
        | ~torch.isfinite(log_mass)
        | (double_values == 0).all(dim=-1)
    ).unsqueeze(-1)
    return torch.where(
        direct_values_are_usable,
        direct_values,
        log_scaled_values,
    )


def _event_group_log_projection_is_required(
    values: Tensor,
    log_weights: Tensor,
    output_gradient: Tensor,
    group_masks: Tensor,
    complement_masks: Tensor,
    group_log_mass: Tensor,
    complement_log_mass: Tensor,
) -> Tensor:
    double_values = values.double()
    double_output_gradient = output_gradient.double()
    nonzero_finite_output = torch.isfinite(double_output_gradient) & (
        double_output_gradient != 0
    )
    minimum_normal_log_magnitude = math.log(torch.finfo(torch.float64).tiny)
    maximum_log_magnitude = math.log(torch.finfo(torch.float64).max)
    nonzero_finite_values = torch.isfinite(double_values) & (double_values != 0)
    relevant_features = nonzero_finite_values & nonzero_finite_output.unsqueeze(1)
    safe_value_magnitudes = torch.where(
        nonzero_finite_values,
        double_values.abs(),
        torch.ones_like(double_values),
    )
    value_log_magnitudes = torch.log(safe_value_magnitudes)
    minimum_value_log_magnitude = torch.where(
        relevant_features,
        value_log_magnitudes,
        torch.full_like(value_log_magnitudes, float("inf")),
    ).amin(dim=-1)
    maximum_value_log_magnitude = torch.where(
        relevant_features,
        value_log_magnitudes,
        torch.full_like(value_log_magnitudes, float("-inf")),
    ).amax(dim=-1)
    has_relevant_feature = relevant_features.any(dim=-1)

    double_log_weights = log_weights.double()
    minimum_weighted_log_magnitude = double_log_weights + minimum_value_log_magnitude
    maximum_weighted_log_magnitude = double_log_weights + maximum_value_log_magnitude
    weighted_value_is_outside_normal_range = has_relevant_feature & (
        (minimum_weighted_log_magnitude < minimum_normal_log_magnitude)
        | (maximum_weighted_log_magnitude > maximum_log_magnitude)
    )

    source_is_relevant = group_masks | complement_masks
    partition_log_scale = torch.where(
        group_masks,
        double_log_weights.unsqueeze(1) + complement_log_mass.unsqueeze(2),
        torch.where(
            complement_masks,
            double_log_weights.unsqueeze(1) + group_log_mass.unsqueeze(2),
            torch.full_like(group_log_mass.unsqueeze(2), float("-inf")),
        ),
    )
    minimum_partition_log_magnitude = (
        partition_log_scale + minimum_value_log_magnitude.unsqueeze(1)
    )
    maximum_partition_log_magnitude = (
        partition_log_scale + maximum_value_log_magnitude.unsqueeze(1)
    )
    partition_term_is_outside_normal_range = (
        torch.isfinite(partition_log_scale)
        & has_relevant_feature.unsqueeze(1)
        & (
            (minimum_partition_log_magnitude < minimum_normal_log_magnitude)
            | (maximum_partition_log_magnitude > maximum_log_magnitude)
        )
    )
    return (
        source_is_relevant
        & torch.isfinite(partition_log_scale)
        & (
            weighted_value_is_outside_normal_range.unsqueeze(1)
            | partition_term_is_outside_normal_range
        )
    ).any(dim=2)


def _stable_event_group_log_projection(
    values: Tensor,
    log_weights: Tensor,
    output_gradient: Tensor,
    group_masks: Tensor,
    complement_masks: Tensor,
    group_log_mass: Tensor,
    complement_log_mass: Tensor,
    required_candidate_mask: Tensor,
) -> Tensor:
    batch_size, beam_width, feature_count = values.shape
    double_values = values.double()
    expanded_output_gradient = (
        output_gradient.double()
        .unsqueeze(1)
        .expand(
            -1,
            beam_width,
            -1,
        )
    )
    fallback_covariance = torch.zeros(
        (batch_size, beam_width),
        dtype=torch.float64,
        device=values.device,
    )
    negative_infinity = torch.full_like(log_weights.double(), float("-inf"))

    for candidate_index in range(beam_width):
        if not bool(required_candidate_mask[:, candidate_index].any().item()):
            continue
        candidate_group_mask = group_masks[:, candidate_index]
        candidate_complement_mask = complement_masks[:, candidate_index]
        term_log_scales = torch.where(
            candidate_group_mask,
            log_weights.double() + complement_log_mass[:, candidate_index].unsqueeze(1),
            torch.where(
                candidate_complement_mask,
                log_weights.double() + group_log_mass[:, candidate_index].unsqueeze(1),
                negative_infinity,
            ),
        )
        term_signs = torch.where(
            candidate_group_mask,
            torch.ones_like(term_log_scales),
            torch.where(
                candidate_complement_mask,
                -torch.ones_like(term_log_scales),
                torch.zeros_like(term_log_scales),
            ),
        )
        signed_output_gradient = expanded_output_gradient * term_signs.unsqueeze(-1)
        expanded_term_log_scales = term_log_scales.unsqueeze(-1).expand(
            -1,
            -1,
            feature_count,
        )
        fallback_covariance[:, candidate_index] = _sum_signed_products_in_log_space(
            double_values.reshape(batch_size, beam_width * feature_count),
            signed_output_gradient.reshape(
                batch_size,
                beam_width * feature_count,
            ),
            expanded_term_log_scales.reshape(
                batch_size,
                beam_width * feature_count,
            ),
        )
    return fallback_covariance


def _stable_event_group_score_gradients(
    values: Tensor,
    log_weights: Tensor,
    output_gradient: Tensor,
    finite_valid_mask: Tensor,
    selected_path_mask: Tensor,
    selected_indices: Tensor,
) -> Tensor:
    batch_size, beam_width = finite_valid_mask.shape
    candidate_selected_mask = selected_path_mask.unsqueeze(2)
    source_selected_mask = selected_path_mask.unsqueeze(1)
    same_score_group_mask = (
        candidate_selected_mask
        & source_selected_mask
        & (selected_indices.unsqueeze(2) == selected_indices.unsqueeze(1))
    )
    group_masks = same_score_group_mask & finite_valid_mask.unsqueeze(1)
    complement_masks = finite_valid_mask.unsqueeze(1) & ~group_masks
    expanded_log_weights = (
        log_weights.double()
        .unsqueeze(1)
        .expand(
            -1,
            beam_width,
            -1,
        )
    )
    partition_shape = (batch_size, beam_width, values.shape[-1])
    group_weighted_sums = torch.zeros(
        partition_shape,
        dtype=torch.float64,
        device=values.device,
    )
    group_sum_compensation = torch.zeros_like(group_weighted_sums)
    complement_weighted_sums = torch.zeros_like(group_weighted_sums)
    complement_sum_compensation = torch.zeros_like(group_weighted_sums)
    weighted_path_values = _scaled_values_from_log_weights(
        values.double(),
        log_weights.double(),
    )
    for source_index in range(beam_width):
        source_contribution = weighted_path_values[
            :, source_index : source_index + 1
        ].expand(
            -1,
            beam_width,
            -1,
        )
        group_contribution = torch.where(
            group_masks[:, :, source_index].unsqueeze(-1),
            source_contribution,
            torch.zeros_like(source_contribution),
        )
        complement_contribution = torch.where(
            complement_masks[:, :, source_index].unsqueeze(-1),
            source_contribution,
            torch.zeros_like(source_contribution),
        )
        group_weighted_sums, group_sum_compensation = _compensated_add(
            group_weighted_sums,
            group_sum_compensation,
            group_contribution,
        )
        complement_weighted_sums, complement_sum_compensation = _compensated_add(
            complement_weighted_sums,
            complement_sum_compensation,
            complement_contribution,
        )
    group_weighted_sums = group_weighted_sums + group_sum_compensation
    complement_weighted_sums = complement_weighted_sums + complement_sum_compensation

    negative_infinity = torch.full_like(expanded_log_weights, float("-inf"))
    group_log_mass = torch.logsumexp(
        torch.where(group_masks, expanded_log_weights, negative_infinity),
        dim=2,
    )
    complement_log_mass = torch.logsumexp(
        torch.where(complement_masks, expanded_log_weights, negative_infinity),
        dim=2,
    )
    direct_weights = torch.exp(log_weights.double()).unsqueeze(1)
    group_mass = _compensated_sum_last_dimension(
        torch.where(
            group_masks,
            direct_weights,
            torch.zeros_like(direct_weights),
        )
    )
    complement_mass = _compensated_sum_last_dimension(
        torch.where(
            complement_masks,
            direct_weights,
            torch.zeros_like(direct_weights),
        )
    )
    scaled_group_sums = _values_times_partition_mass(
        group_weighted_sums,
        complement_log_mass,
        complement_mass,
    )
    scaled_complement_sums = _values_times_partition_mass(
        complement_weighted_sums,
        group_log_mass,
        group_mass,
    )
    # Once both partition accumulations are finite, each scaled partition is
    # bounded by m(1-m) * max(abs(values)). Their difference is therefore at
    # most 2m(1-m) <= 1/2 of that finite maximum and cannot overflow. A rounded
    # partition accumulation that already overflowed is redirected below.
    partition_differences, partition_difference_roundoff = _difference_with_roundoff(
        scaled_group_sums,
        scaled_complement_sums,
    )
    expanded_output_gradient = (
        output_gradient.double().unsqueeze(1).expand_as(partition_differences)
    )
    ordinary_covariance = _dot_product_times_log_scale(
        partition_differences,
        expanded_output_gradient,
        torch.zeros_like(group_log_mass),
        torch.ones_like(group_mass),
    )
    roundoff_covariance = _dot_product_times_log_scale(
        partition_difference_roundoff,
        expanded_output_gradient,
        torch.zeros_like(group_log_mass),
        torch.ones_like(group_mass),
    )
    covariance = _compensated_sum_last_dimension(
        torch.stack((ordinary_covariance, roundoff_covariance), dim=-1)
    )
    all_valid_values_are_finite = (
        ~finite_valid_mask.unsqueeze(-1) | torch.isfinite(values)
    ).all(dim=(1, 2))
    partition_accumulation_is_nonfinite = (
        (
            ~torch.isfinite(group_weighted_sums)
            | ~torch.isfinite(complement_weighted_sums)
        ).any(dim=-1)
        & torch.isfinite(group_log_mass)
        & torch.isfinite(complement_log_mass)
        & all_valid_values_are_finite.unsqueeze(1)
    )
    prior_candidate_mask = torch.tril(
        torch.ones(
            (beam_width, beam_width),
            dtype=torch.bool,
            device=values.device,
        ),
        diagonal=-1,
    )
    first_group_candidate_mask = ~(
        same_score_group_mask & prior_candidate_mask.unsqueeze(0)
    ).any(dim=2)
    log_projection_required_mask = (
        (
            _event_group_log_projection_is_required(
                values,
                log_weights,
                output_gradient,
                group_masks,
                complement_masks,
                group_log_mass,
                complement_log_mass,
            )
            | partition_accumulation_is_nonfinite
        )
        & selected_path_mask
        & first_group_candidate_mask
    )
    if bool(log_projection_required_mask.any().item()):
        log_projection_covariance = _stable_event_group_log_projection(
            values,
            log_weights,
            output_gradient,
            group_masks,
            complement_masks,
            group_log_mass,
            complement_log_mass,
            log_projection_required_mask,
        )
        covariance = torch.where(
            log_projection_required_mask,
            log_projection_covariance,
            covariance,
        )

    return torch.where(
        selected_path_mask & first_group_candidate_mask,
        covariance,
        torch.zeros_like(covariance),
    )


def _compensated_event_score_gradient(
    values: Tensor,
    log_weights: Tensor,
    output_gradient: Tensor,
    finite_valid_mask: Tensor,
    router_scores: Tensor,
    flattened_selected_indices: Tensor,
) -> Tensor:
    batch_size, beam_width = finite_valid_mask.shape
    selected_indices = flattened_selected_indices.reshape(batch_size, beam_width)
    if router_scores.numel() == 0:
        return torch.zeros_like(router_scores)
    if router_scores.numel() % batch_size != 0:
        raise ValueError(
            "Every beam router-score event must divide evenly across the batch."
        )

    score_cells_per_batch = router_scores.numel() // batch_size
    batch_offsets = (
        torch.arange(batch_size, device=router_scores.device) * score_cells_per_batch
    ).unsqueeze(1)
    local_selected_indices = selected_indices - batch_offsets
    selected_index_in_range_mask = (local_selected_indices >= 0) & (
        local_selected_indices < score_cells_per_batch
    )
    safe_selected_indices = local_selected_indices.clamp(
        min=0,
        max=score_cells_per_batch - 1,
    )
    batched_router_scores = router_scores.reshape(batch_size, score_cells_per_batch)
    selected_router_scores = batched_router_scores.gather(1, safe_selected_indices)
    selected_path_mask = (
        finite_valid_mask
        & selected_index_in_range_mask
        & torch.isfinite(selected_router_scores)
    )
    group_score_gradients = _stable_event_group_score_gradients(
        values,
        log_weights,
        output_gradient,
        finite_valid_mask,
        selected_path_mask,
        safe_selected_indices,
    )
    accumulated_gradient = torch.zeros(
        (batch_size, score_cells_per_batch),
        dtype=torch.float64,
        device=router_scores.device,
    )
    accumulated_compensation = torch.zeros_like(accumulated_gradient)
    for beam_index in range(beam_width):
        beam_gradient = torch.zeros_like(accumulated_gradient).scatter(
            1,
            safe_selected_indices[:, beam_index : beam_index + 1],
            group_score_gradients[:, beam_index : beam_index + 1],
        )
        accumulated_gradient, accumulated_compensation = _compensated_add(
            accumulated_gradient,
            accumulated_compensation,
            beam_gradient,
        )
    return (
        (accumulated_gradient + accumulated_compensation)
        .to(router_scores.dtype)
        .reshape_as(router_scores)
    )


class _StableBeamMixtureWithScoreHistory(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        values: Tensor,
        scores: Tensor,
        valid_mask: Tensor,
        *score_history_data: Tensor,
    ) -> Tensor:
        event_count = len(score_history_data) // 2
        router_score_events = score_history_data[:event_count]
        selected_score_indices = score_history_data[event_count:]
        working_scores = (
            scores.float()
            if scores.dtype in (torch.float16, torch.bfloat16)
            else scores
        )
        conditional_log_weights, finite_valid_mask = _conditional_log_route_weights(
            working_scores,
            valid_mask,
        )
        ctx.event_count = event_count
        ctx.save_for_backward(
            values,
            conditional_log_weights,
            finite_valid_mask,
            *router_score_events,
            *selected_score_indices,
        )
        return _stable_weighted_average(values, conditional_log_weights).to(
            values.dtype
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        output_gradient: Tensor,
    ) -> tuple[Tensor | None, ...]:
        saved_tensors = ctx.saved_tensors
        values, log_weights, finite_valid_mask = saved_tensors[:3]
        event_count = ctx.event_count
        router_score_events = saved_tensors[3 : 3 + event_count]
        selected_score_indices = saved_tensors[3 + event_count :]

        expanded_output_gradient = output_gradient.unsqueeze(1).expand_as(values)
        value_gradients = _scaled_values_from_log_weights(
            expanded_output_gradient,
            log_weights,
        ).to(values.dtype)

        # Each event groups final paths by the raw score cell in their ancestry.
        # A group-versus-complement covariance cancels shared prefixes before
        # incompatible cotangents are compressed. Peak memory is
        # O(batch * (beam_width**2 + beam_width * features)); history work is
        # ordinarily O(events * batch * beam_width**2 * features), independent
        # of score capacity. If a weighted intermediate leaves float64's normal
        # range, the exact-product fallback retains that peak-memory bound but
        # can take O(events * batch * beam_width**3 * features**2) time.
        event_gradients = []
        for router_scores, flattened_selected_indices in zip(
            router_score_events,
            selected_score_indices,
            strict=True,
        ):
            event_gradients.append(
                _compensated_event_score_gradient(
                    values,
                    log_weights,
                    output_gradient,
                    finite_valid_mask,
                    router_scores,
                    flattened_selected_indices,
                )
            )

        return (
            value_gradients,
            None,
            None,
            *event_gradients,
            *([None] * event_count),
        )


def _stable_beam_mixture_with_score_history(
    values: Tensor,
    scores: Tensor,
    valid_mask: Tensor,
    router_score_events: tuple[Tensor, ...],
    selected_score_indices: tuple[Tensor, ...],
) -> Tensor:
    if len(router_score_events) != len(selected_score_indices):
        raise ValueError("Beam score events and ancestry indices must stay aligned.")
    expected_beam_slot_count = values.shape[0] * values.shape[1]
    if any(
        score_indices.numel() != expected_beam_slot_count
        for score_indices in selected_score_indices
    ):
        raise ValueError(
            "Every beam score ancestry event must identify each final beam slot."
        )
    if any(
        score_indices.dtype not in (torch.int32, torch.int64)
        for score_indices in selected_score_indices
    ):
        raise TypeError("Beam score ancestry indices must use an integer dtype.")
    return _StableBeamMixtureWithScoreHistory.apply(
        values,
        scores,
        valid_mask,
        *router_score_events,
        *selected_score_indices,
    )


class _StableUnnormalizedWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        values: Tensor,
        log_weights: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(values, log_weights)
        ctx.log_weights_dtype = log_weights.dtype
        return (
            _scaled_values_from_log_weights(values, log_weights)
            .sum(dim=1)
            .to(values.dtype)
        )

    @staticmethod
    def backward(
        ctx: Any,
        output_gradient: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values, log_weights = ctx.saved_tensors
        expanded_output_gradient = output_gradient.unsqueeze(1).expand_as(values)
        value_gradients = _scaled_values_from_log_weights(
            expanded_output_gradient,
            log_weights,
        ).to(values.dtype)
        log_weight_gradients = _dot_product_times_log_scale(
            values,
            expanded_output_gradient,
            log_weights,
        )
        return (
            value_gradients,
            log_weight_gradients.to(ctx.log_weights_dtype),
        )


def _signed_log_weighted_sum(values: Tensor, log_weights: Tensor) -> Tensor:
    return _StableUnnormalizedWeightedSum.apply(values, log_weights)


def weighted_branch_candidate(
    branch_outputs: Tensor,
    probabilities: Tensor,
    valid_branch_mask: Tensor,
    *,
    log_probabilities: Tensor | None = None,
    router_scores: Tensor | None = None,
) -> Tensor:
    normalization_probabilities = (
        probabilities.float()
        if probabilities.dtype in (torch.float16, torch.bfloat16)
        else probabilities
    )
    valid_weights = normalization_probabilities * valid_branch_mask.to(
        normalization_probabilities.dtype
    )
    valid_weight_sums = valid_weights.sum(dim=1, keepdim=True)
    # With no valid branches every row of branch_outputs holds the
    # unprocessed input, so falling back to the original probabilities
    # passes the input through scaled by their sum — exactly the input
    # only when the sampler normalizes probabilities to 1 — while
    # keeping the gradient path through the router.
    row_has_valid_weight = valid_weight_sums > 0
    safe_weight_sums = torch.where(
        row_has_valid_weight,
        valid_weight_sums,
        torch.ones_like(valid_weight_sums),
    )
    normalized_valid_weights = valid_weights / safe_weight_sums
    single_valid_branch_mask = (
        valid_branch_mask.sum(dim=1, keepdim=True) == 1
    ) & row_has_valid_weight
    # A normalized single branch is exactly one in the forward pass, so
    # the quotient's true Jacobian is zero and a top-1 router could never
    # learn from the task loss. Preserve that established forward value
    # while using the bounded Jacobian of the unnormalized probability.
    # Selection and validity remain discrete; only the selected
    # probability receives this straight-through gradient.
    single_branch_surrogate = normalized_valid_weights + (
        valid_weights - valid_weights.detach()
    )
    valid_route_weights = torch.where(
        single_valid_branch_mask,
        single_branch_surrogate,
        normalized_valid_weights,
    )
    effective_weights = torch.where(
        row_has_valid_weight,
        valid_route_weights,
        normalization_probabilities,
    ).to(probabilities.dtype)
    if log_probabilities is not None:
        stable_log_probabilities = (
            log_probabilities.float()
            if log_probabilities.dtype in (torch.float16, torch.bfloat16)
            else log_probabilities
        )
        (
            stable_forward_weights,
            surrogate_weights,
            finite_valid_mask,
        ) = _log_space_route_weights(
            stable_log_probabilities,
            valid_branch_mask,
        )
        subnormal_valid_probability_mask = finite_valid_mask & (
            probabilities.abs() < torch.finfo(probabilities.dtype).tiny
        )
        has_finite_difference_overflow = _row_has_finite_difference_overflow(
            branch_outputs,
            finite_valid_mask,
        )
        has_equal_weight_cancellation = _row_has_equal_weight_cancellation(
            branch_outputs,
            stable_log_probabilities,
            finite_valid_mask,
        )
        stable_forward_weight_row_mask = (
            subnormal_valid_probability_mask.any(dim=1, keepdim=True)
            | has_finite_difference_overflow
            | has_equal_weight_cancellation
        )
        effective_weights = torch.where(
            stable_forward_weight_row_mask,
            stable_forward_weights.to(effective_weights.dtype),
            effective_weights,
        )
        effective_weights = _forward_value_with_surrogate_gradient(
            effective_weights,
            surrogate_weights,
        )
        regular_candidate = (branch_outputs * effective_weights.unsqueeze(-1)).sum(
            dim=1
        )
        working_log_probabilities = log_probabilities
        _, stable_finite_mask = _conditional_log_route_weights(
            working_log_probabilities,
            valid_branch_mask,
        )
        stable_gradient_scores = (
            working_log_probabilities if router_scores is None else router_scores
        )
        stable_candidate = _stable_weighted_sum(
            branch_outputs,
            stable_gradient_scores,
            stable_finite_mask,
        )
        equal_score_translation_risk_mask = (
            _row_has_equal_valid_scores(
                stable_log_probabilities,
                stable_finite_mask,
            )
            & _row_has_opposing_centered_values(
                branch_outputs,
                stable_candidate,
                stable_finite_mask,
            )
            & (
                regular_candidate.detach()
                != stable_candidate.detach().to(regular_candidate.dtype)
            ).any(dim=1, keepdim=True)
        )
        stable_forward_weight_row_mask = (
            stable_forward_weight_row_mask | equal_score_translation_risk_mask
        )
        finite_branch_count = stable_finite_mask.sum(dim=1, keepdim=True)
        stable_multi_branch_candidate = stable_candidate
        single_finite_branch_mask = finite_branch_count == 1
        single_branch_forward = (
            branch_outputs * stable_finite_mask.to(branch_outputs.dtype).unsqueeze(-1)
        ).sum(dim=1)
        unconditional_valid_log_weights = torch.where(
            stable_finite_mask,
            working_log_probabilities,
            torch.full_like(
                working_log_probabilities,
                float("-inf"),
            ),
        )
        single_branch_router_surrogate = _signed_log_weighted_sum(
            branch_outputs.detach(),
            unconditional_valid_log_weights,
        )
        single_branch_candidate = single_branch_forward + (
            single_branch_router_surrogate - single_branch_router_surrogate.detach()
        )
        stable_forward_candidate = torch.where(
            single_finite_branch_mask,
            single_branch_candidate,
            stable_multi_branch_candidate,
        )
        stable_gradient_candidate = torch.where(
            finite_branch_count >= 2,
            stable_multi_branch_candidate,
            regular_candidate,
        )
        regular_forward_candidate = _forward_value_with_surrogate_gradient(
            regular_candidate,
            stable_gradient_candidate,
        )
        return torch.where(
            stable_forward_weight_row_mask,
            stable_forward_candidate,
            regular_forward_candidate,
        )
    return (branch_outputs * effective_weights.unsqueeze(-1)).sum(dim=1)
