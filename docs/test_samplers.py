import torch
import unittest
from math import prod

from emperor.sampler.model import SamplerModel
from emperor.sampler.core.config import SamplerConfig
from emperor.sampler.core.base import SamplerBase
from emperor.sampler.core.variants import SamplerFull, SamplerSparse, SamplerTopk


class SamplerTestCase(unittest.TestCase):
    def preset(
        self,
        top_k: int = 3,
        threshold: float = 0.0,
        filter_above_threshold: bool = False,
        num_topk_samples: int = 0,
        normalize_probabilities_flag: bool = False,
        noisy_topk_flag: bool = False,
        num_experts: int = 5,
        coefficient_of_variation_loss_weight: float = 0.0,
        switch_loss_weight: float = 0.0,
        zero_centred_loss_weight: float = 0.0,
        mutual_information_loss_weight: float = 0.0,
    ) -> SamplerConfig:
        return SamplerConfig(
            top_k=top_k,
            threshold=threshold,
            filter_above_threshold=filter_above_threshold,
            num_topk_samples=num_topk_samples,
            normalize_probabilities_flag=normalize_probabilities_flag,
            noisy_topk_flag=noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
            switch_loss_weight=switch_loss_weight,
            zero_centred_loss_weight=zero_centred_loss_weight,
            mutual_information_loss_weight=mutual_information_loss_weight,
            router_config=None,
        )


class TestProbabilitySampler(SamplerTestCase):
    def test_init(self):
        cfg = self.preset()
        c = cfg
        m = SamplerBase(cfg)

        self.assertEqual(m.top_k, c.top_k)
        self.assertEqual(m.threshold, c.threshold)
        self.assertEqual(m.num_topk_samples, c.num_topk_samples)
        self.assertEqual(m.noisy_topk_flag, c.noisy_topk_flag)
        self.assertEqual(m.num_experts, c.num_experts)
        self.assertEqual(
            m.normalize_probabilities_flag,
            c.normalize_probabilities_flag,
        )

    def test_normalize_probabilities__normalize_probabilities_flag__False(self):
        normalize_probs = [False, True]
        for flag in normalize_probs:
            message = f"Testing configuration with normalize_probabilities_flag={flag}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    normalize_probabilities_flag=flag,
                )
                m = SamplerBase(cfg)
                probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
                result = m._normalize_probabilities(probs)

                if flag:
                    distribution_check = result.sum(dim=-1)
                    expected_distribution = torch.tensor([1.0, 1.0])
                    self.assertTrue(
                        torch.allclose(
                            distribution_check.round(decimals=4),
                            expected_distribution.round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )
                else:
                    self.assertTrue(
                        torch.allclose(
                            probs.round(decimals=4),
                            result.round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )

    def test__probability_sampling_strategy(self):
        cfg = self.preset()
        m = SamplerBase(cfg)
        probabilities = torch.randn(2, 4)

        with self.assertRaises(NotImplementedError) as context:
            m._sample_probabilities_and_indices(probabilities)

    def test__update_mask_given_threshold__threshold__zero(self):
        threshold_options = [0.0, 0.4]
        filter_above_threshold_options = [False, True]

        for threshold_option in threshold_options:
            for filter_flag in filter_above_threshold_options:
                message = (
                    f"Testing configuration with threshold={threshold_option}, "
                    f"filter_above_threshold={filter_flag}"
                )
                with self.subTest(msg=message):
                    cfg = self.preset(
                        threshold=threshold_option,
                        filter_above_threshold=filter_flag,
                    )
                    m = SamplerBase(cfg)

                    batch_size = 2
                    skip_mask = torch.ones(batch_size).reshape(-1, 1)
                    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
                    output = m._SamplerBase__update_mask_given_threshold(
                        probabilities, skip_mask
                    )
                    expected_mask = torch.ones(batch_size).reshape(-1, 1)
                    if threshold_option > 0.0:
                        if filter_flag:
                            expected_mask[0] = 0.0
                            self.assertTrue(
                                torch.allclose(
                                    output.round(decimals=4),
                                    expected_mask.round(decimals=4),
                                    atol=1e-6,
                                    rtol=1e-5,
                                )
                            )
                        else:
                            expected_mask[1] = 0.0
                            self.assertTrue(
                                torch.allclose(
                                    output.round(decimals=4),
                                    expected_mask.round(decimals=4),
                                    atol=1e-6,
                                    rtol=1e-5,
                                )
                            )
                    else:
                        self.assertTrue(
                            torch.allclose(
                                skip_mask.round(decimals=4),
                                output.round(decimals=4),
                                atol=1e-6,
                                rtol=1e-5,
                            )
                        )

    def test__apply_skip_mask__threshold__zero(self):
        threshold_options = [0.0, 0.4]

        for threshold_option in threshold_options:
            message = f"Testing configuration with threshold={threshold_option}, "
            with self.subTest(msg=message):
                cfg = self.preset(
                    threshold=threshold_option,
                )
                m = SamplerBase(cfg)

                batch_size = 2
                sequence_length = 3
                feature_dim = 4
                probs = torch.ones(batch_size * sequence_length, feature_dim)
                logits = torch.ones(batch_size * sequence_length, feature_dim)
                mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)
                unmasked_token = 0
                mask[unmasked_token, :] = 1
                masked_probs, router_logit_scores = m._SamplerBase__apply_skip_mask(
                    probs, logits, mask
                )

                if threshold_option == 0.0:
                    self.assertTrue(
                        torch.allclose(
                            masked_probs.round(decimals=4),
                            probs.round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            router_logit_scores.round(decimals=4),
                            logits.round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )
                else:
                    self.assertTrue(
                        torch.sum(masked_probs[unmasked_token, :]).item() > 0
                    )
                    self.assertTrue(torch.sum(masked_probs[1:, :]).item() == 0)
                    self.assertTrue(
                        torch.sum(router_logit_scores[unmasked_token, :]).item() > 0
                    )
                    self.assertTrue(torch.sum(router_logit_scores[1:, :]).item() == 0)

    def test__add_noise_to_logits_flag__False(self):
        noisy_topk_flag_options = [True, False]

        for noisy_topk_flag_option in noisy_topk_flag_options:
            message = (
                f"Testing configuration with noisy_topk_flag={noisy_topk_flag_option}"
            )
            with self.subTest(msg=message):
                cfg = self.preset(
                    noisy_topk_flag=noisy_topk_flag_option,
                )
                m = SamplerBase(cfg)
                m.training = noisy_topk_flag_option

                batch_size = 2
                shape = (batch_size, m.cfg.num_experts * 2)
                logits = torch.arange(prod(shape)).reshape(shape).float()
                result = m._SamplerBase__add_noise_to_logits(logits)

                if noisy_topk_flag_option:
                    self.assertNotEqual(logits.mean(), result.mean())
                    self.assertEqual(result.shape, (batch_size, m.num_experts))
                else:
                    self.assertTrue(
                        torch.allclose(
                            logits.round(decimals=4),
                            result.round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )

    def test__add_noise_to_logits_uses_first_half_without_noise_in_eval(self):
        cfg = self.preset(noisy_topk_flag=True, num_experts=5)
        m = SamplerBase(cfg)
        m.eval()
        logits = torch.tensor(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                [0.5, 1.5, 2.5, 3.5, 4.5, 9.0, 8.0, 7.0, 6.0, 5.0],
            ]
        )

        result = m._SamplerBase__add_noise_to_logits(logits)

        torch.testing.assert_close(result, logits[:, : m.num_experts])

    def test__compute_masked_probabilities(self):
        noisy_topk_flag_options = [True, False]
        threshold_options = [0.0, 0.2]
        for threshold_option in threshold_options:
            for noisy_topk_flag_option in noisy_topk_flag_options:
                message = (
                    f"Testing configuration with threshold={threshold_option}, "
                    f"noisy_topk_flag_option={noisy_topk_flag_option}"
                )
                with self.subTest(msg=message):
                    cfg = self.preset(
                        noisy_topk_flag=noisy_topk_flag_option,
                        threshold=threshold_option,
                    )
                    m = SamplerBase(cfg)

                    batch_size = 3
                    sequence_length = 4
                    num_expert_dim = (
                        m.num_experts * 2 if noisy_topk_flag_option else m.num_experts
                    )
                    shape = (batch_size * sequence_length, num_expert_dim)
                    router_logit_scores = (
                        torch.arange(prod(shape)).reshape(shape).float()
                    )

                    mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)
                    unmasked_token = 0
                    mask[unmasked_token, :] = 1

                    masked_probabilities, router_logit_scores = (
                        m._SamplerBase__compute_masked_probabilities(
                            router_logit_scores, mask
                        )
                    )

                    if noisy_topk_flag_option and (threshold_option > 0.0):
                        self.assertEqual(
                            masked_probabilities.shape,
                            (batch_size * sequence_length, m.cfg.num_experts),
                        )
                        self.assertTrue(
                            torch.sum(masked_probabilities[0, :]).item() > 0
                        )
                        self.assertTrue(
                            torch.sum(masked_probabilities[1:, :]).item() == 0
                        )
                        self.assertTrue(torch.sum(router_logit_scores[0, :]).item() > 0)
                        self.assertTrue(
                            torch.sum(router_logit_scores[1:, :]).item() == 0
                        )
                        self.assertTrue(
                            torch.allclose(
                                torch.sum(masked_probabilities[0, :], dim=-1).round(
                                    decimals=4
                                ),
                                torch.tensor(1.0).float().round(decimals=4),
                                atol=1e-6,
                                rtol=1e-5,
                            )
                        )
                    elif noisy_topk_flag_option and (threshold_option == 0.0):
                        self.assertEqual(
                            masked_probabilities.shape,
                            (batch_size * sequence_length, m.num_experts),
                        )
                    else:
                        self.assertEqual(
                            masked_probabilities.shape,
                            router_logit_scores.shape,
                        )

    def test_get_probabilities_and_indices_rejects_invalid_inputs(self):
        cfg = self.preset(top_k=2, num_experts=4)
        m = SamplerTopk(cfg)

        invalid_logits = [
            [1.0, 2.0, 3.0, 4.0],
            torch.ones(4),
            torch.ones(2, 3, 4),
            torch.ones(2, 5),
        ]
        for logits in invalid_logits:
            with self.subTest(logits=logits):
                with self.assertRaises((TypeError, ValueError)):
                    m.get_probabilities_and_indices(logits)

        with self.assertRaises(TypeError):
            m.get_probabilities_and_indices(torch.ones(2, 4), skip_mask=[1, 1])

        with self.assertRaises(ValueError):
            m.get_probabilities_and_indices(
                torch.ones(2, 4),
                skip_mask=torch.ones(3, 1),
            )

    def test_get_probabilities_and_indices_stores_loss_and_skip_mask(self):
        cfg = self.preset(
            top_k=2,
            num_experts=4,
            threshold=0.2,
            filter_above_threshold=False,
        )
        m = SamplerTopk(cfg)
        logits = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [-4.0, -3.0, -2.0, -1.0],
            ]
        )
        skip_mask = torch.ones(2, 1)

        _, _, updated_skip_mask, loss = m.get_probabilities_and_indices(
            logits, skip_mask
        )

        self.assertIs(m.updated_skip_mask, updated_skip_mask)
        self.assertIs(m.auxiliary_loss, loss)


class TestSamplerSparse(SamplerTestCase):
    def test_sample_probabilities_and_indices(self):
        cfg = self.preset()
        m = SamplerSparse(cfg)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        probability, indices = m._sample_probabilities_and_indices(probabilities)
        self.assertEqual(probability.shape, (batch_size,))
        self.assertEqual(indices.shape, (batch_size,))

    def test_sample_probabilities_and_indices_selects_max_deterministically(self):
        cfg = self.preset()
        m = SamplerSparse(cfg)
        probabilities = torch.tensor(
            [
                [0.1, 0.7, 0.2, 0.0, 0.0],
                [0.4, 0.1, 0.3, 0.2, 0.0],
            ]
        )

        probability, indices = m._sample_probabilities_and_indices(probabilities)

        torch.testing.assert_close(probability, torch.tensor([0.7, 0.4]))
        torch.testing.assert_close(indices, torch.tensor([1, 0]))

    def test_get_probabilities_and_indices(self):
        loss_options = [0.0, 0.1]
        noisy_flag_options = [True, False]
        threshold_options = [0.0, 0.2]
        filter_above_threshold_options = [True, False]
        normalize_probabilities_flag = [True, False]

        for loss in loss_options:
            for noisy_flag in noisy_flag_options:
                for thresh in threshold_options:
                    for filter_flag in filter_above_threshold_options:
                        for normalize_flag in normalize_probabilities_flag:
                            message = (
                                f"Testing configuration with loss={loss}, "
                                f"noisy_flag={noisy_flag}, "
                                f"threshold={thresh}, "
                                f"filter_flag={filter_flag}, "
                                f"normalize_flag={normalize_flag}"
                            )
                            with self.subTest(msg=message):
                                if normalize_flag:
                                    with self.assertRaises(ValueError):
                                        cfg = self.preset(
                                            normalize_probabilities_flag=normalize_flag,
                                        )
                                        m = SamplerSparse(cfg)
                                else:
                                    cfg = self.preset(
                                        noisy_topk_flag=noisy_flag,
                                        threshold=thresh,
                                        filter_above_threshold=filter_flag,
                                        coefficient_of_variation_loss_weight=loss,
                                        switch_loss_weight=loss,
                                        zero_centred_loss_weight=loss,
                                        # This is not work for SamplerSparse
                                        # mutual_information_loss_weight=loss,
                                    )
                                    m = SamplerSparse(cfg)

                                    batch_size = 3
                                    sequence_length = 4
                                    router_output_dim = (
                                        m.num_experts * 2
                                        if noisy_flag
                                        else m.num_experts
                                    )
                                    shape = (
                                        batch_size * sequence_length,
                                        router_output_dim,
                                    )
                                    logits = torch.softmax(torch.randn(*shape), dim=-1)
                                    mask = torch.ones(
                                        batch_size, sequence_length
                                    ).reshape(-1, 1)
                                    probabilities, selected_indices, skip_mask, sampler_loss = (
                                        m.get_probabilities_and_indices(logits, mask)
                                    )
                                    self.assertEqual(
                                        probabilities.shape,
                                        (batch_size * sequence_length,),
                                    )
                                    self.assertEqual(
                                        selected_indices.shape,
                                        (batch_size * sequence_length,),
                                    )
                                    if sampler_loss > 0:
                                        self.assertTrue(sampler_loss >= 0)

    def test__prepare_loss_skip_mask(self):
        input_options = [None, torch.ones(3, 4)]
        for input_mask in input_options:
            message = f"Testing configuration with skip_mask={input_mask}"
            with self.subTest(msg=message):
                cfg = self.preset()
                m = SamplerSparse(cfg)

                skip_maks = m._SamplerSparse__prepare_loss_skip_mask(input_mask)

                if input_mask is None:
                    self.assertIsNone(skip_maks)
                else:
                    self.assertTrue(
                        torch.allclose(
                            skip_maks.round(decimals=4),
                            input_mask.reshape(-1, 1).round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )

    def test__prepare_loss_gates(self):
        cfg = self.preset(
            top_k=1,
        )
        m = SamplerSparse(cfg)

        sampled_probabilities = torch.tensor([0.9, 0.4, 0.7])
        indices = torch.tensor([[2], [0], [3]])

        gates = m._SamplerSparse__prepare_loss_gates(sampled_probabilities, indices)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.9, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7, 0.0],
            ]
        )
        self.assertEqual(gates.shape, (3, m.num_experts))
        torch.testing.assert_close(gates, expected)

    def test_compute_loss(self):
        loss_options = [0.0, 0.1]
        for cross_option in loss_options:
            for coeff_option in loss_options:
                for zero_option in loss_options:
                    message = f"Running test with cross_option={cross_option}, coeff_option={coeff_option}, zero_option={zero_option}"
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            top_k=1,
                            noisy_topk_flag=True,
                            threshold=0.8,
                            filter_above_threshold=True,
                            normalize_probabilities_flag=False,
                            coefficient_of_variation_loss_weight=coeff_option,
                            switch_loss_weight=cross_option,
                            zero_centred_loss_weight=zero_option,
                            # mutual_information_loss_weight=mutual_option,
                        )
                        m = SamplerSparse(cfg)

                        batch_size = 3
                        sequence_length = 4
                        shape = (batch_size * sequence_length, m.num_experts)

                        logits = torch.randn(*shape)
                        full_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
                        sampled_probabilities = torch.rand(
                            batch_size * sequence_length
                        )
                        indices = torch.randint(
                            0, m.num_experts, (batch_size * sequence_length, m.top_k)
                        )
                        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
                        unmasked_token = 0
                        mask[unmasked_token, :] = 1

                        total_loss = m._compute_loss(
                            logits,
                            full_probabilities,
                            sampled_probabilities,
                            indices,
                            mask,
                        )

                        if (
                            coeff_option == 0.0
                            and cross_option == 0.0
                            and zero_option == 0.0
                        ):
                            self.assertTrue(total_loss == 0)
                        else:
                            self.assertTrue(total_loss > 0)
                        self.assertIsInstance(total_loss, torch.Tensor)


class TestSamplerTopk(SamplerTestCase):
    def test_sample_probabilities_and_indices(self):
        cfg = self.preset()
        m = SamplerTopk(cfg)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        probability, indices = m._sample_probabilities_and_indices(probabilities)
        self.assertEqual(probability.shape, (batch_size, m.top_k))
        self.assertEqual(indices.shape, (batch_size, m.top_k))

    def test_sample_probabilities_and_indices_selects_topk_deterministically(self):
        cfg = self.preset(top_k=3, num_experts=5)
        m = SamplerTopk(cfg)
        probabilities = torch.tensor(
            [
                [0.1, 0.7, 0.2, 0.0, 0.4],
                [0.4, 0.1, 0.3, 0.2, 0.0],
            ]
        )

        probability, indices = m._sample_probabilities_and_indices(probabilities)

        torch.testing.assert_close(
            probability,
            torch.tensor(
                [
                    [0.7, 0.4, 0.2],
                    [0.4, 0.3, 0.2],
                ]
            ),
        )
        torch.testing.assert_close(
            indices,
            torch.tensor(
                [
                    [1, 4, 2],
                    [0, 2, 3],
                ]
            ),
        )

    def test_sample_probabilities_and_indices_uses_random_topk_training_branch(self):
        cfg = self.preset(top_k=3, num_experts=5, num_topk_samples=1)
        m = SamplerTopk(cfg)
        m.train()
        probabilities = torch.tensor(
            [
                [0.5, 0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ]
        )
        torch.manual_seed(7)

        probability, indices = m._sample_probabilities_and_indices(probabilities)

        self.assertEqual(probability.shape, (2, 3))
        self.assertEqual(indices.shape, (2, 3))
        torch.testing.assert_close(indices[:, :2], torch.tensor([[0, 1], [4, 3]]))
        self.assertFalse(torch.any(indices[0, 2:] == 0))
        self.assertFalse(torch.any(indices[0, 2:] == 1))
        self.assertFalse(torch.any(indices[1, 2:] == 4))
        self.assertFalse(torch.any(indices[1, 2:] == 3))
        torch.testing.assert_close(probability, torch.gather(probabilities, 1, indices))

    def test_sample_probabilities_and_indices_uses_deterministic_topk_in_eval(self):
        cfg = self.preset(top_k=3, num_experts=5, num_topk_samples=1)
        m = SamplerTopk(cfg)
        m.eval()
        probabilities = torch.tensor(
            [
                [0.5, 0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ]
        )

        probability, indices = m._sample_probabilities_and_indices(probabilities)

        torch.testing.assert_close(
            probability,
            torch.tensor(
                [
                    [0.5, 0.4, 0.3],
                    [0.5, 0.4, 0.3],
                ]
            ),
        )
        torch.testing.assert_close(indices, torch.tensor([[0, 1, 2], [4, 3, 2]]))

    def test_get_probabilities_and_indices(self):
        loss_options = [0.0, 0.1]
        noisy_flag_options = [True, False]
        threshold_options = [0.0, 0.2]
        filter_above_threshold_options = [True, False]
        normalize_probabilities_flag = [True, False]

        for loss in loss_options:
            for noisy_flag in noisy_flag_options:
                for thresh in threshold_options:
                    for filter_flag in filter_above_threshold_options:
                        for normalize_flag in normalize_probabilities_flag:
                            message = (
                                f"Testing configuration with loss={loss}, "
                                f"noisy_flag={noisy_flag}, "
                                f"threshold={thresh}, "
                                f"filter_flag={filter_flag}, "
                                f"normalize_flag={normalize_flag}"
                            )
                            with self.subTest(msg=message):
                                cfg = self.preset(
                                    top_k=3,
                                    noisy_topk_flag=noisy_flag,
                                    threshold=thresh,
                                    filter_above_threshold=filter_flag,
                                    coefficient_of_variation_loss_weight=loss,
                                    switch_loss_weight=loss,
                                    zero_centred_loss_weight=loss,
                                    mutual_information_loss_weight=loss,
                                )
                                m = SamplerTopk(cfg)

                                batch_size = 3
                                sequence_length = 4
                                num_expert_dim = m.num_experts
                                if noisy_flag:
                                    num_expert_dim = m.num_experts * 2
                                shape = (
                                    batch_size * sequence_length,
                                    num_expert_dim,
                                )
                                logits = torch.softmax(torch.randn(*shape), dim=-1)
                                mask = torch.ones(batch_size, sequence_length).reshape(
                                    -1, 1
                                )
                                probabilities, selected_indices, skip_mask, sampler_loss = (
                                    m.get_probabilities_and_indices(logits, mask)
                                )
                                self.assertEqual(
                                    probabilities.shape,
                                    (batch_size * sequence_length, m.top_k),
                                )
                                self.assertEqual(
                                    selected_indices.shape,
                                    (batch_size * sequence_length, m.top_k),
                                )
                                if sampler_loss > 0:
                                    self.assertTrue(sampler_loss >= 0)

    def test__prepare_loss_skip_mask(self):
        input_options = [None, torch.ones(3, 4)]
        for input_mask in input_options:
            message = f"Testing configuration with skip_mask={input_mask}"
            with self.subTest(msg=message):
                cfg = self.preset()
                m = SamplerTopk(cfg)

                skip_maks = m._SamplerTopk__prepare_loss_skip_mask(input_mask)
                if input_mask is None:
                    self.assertIsNone(skip_maks)
                else:
                    self.assertTrue(
                        torch.allclose(
                            skip_maks.round(decimals=4),
                            input_mask.reshape(-1, 1).round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )

    def test__prepare_loss_gates(self):
        cfg = self.preset(top_k=2)
        m = SamplerTopk(cfg)

        sampled_probabilities = torch.tensor(
            [
                [0.7, 0.2],
                [0.3, 0.4],
            ]
        )
        indices = torch.tensor(
            [
                [1, 3],
                [0, 2],
            ]
        )

        gates = m._SamplerTopk__prepare_loss_gates(sampled_probabilities, indices)

        expected = torch.tensor(
            [
                [0.0, 0.7, 0.0, 0.2, 0.0],
                [0.3, 0.0, 0.4, 0.0, 0.0],
            ]
        )
        self.assertEqual(gates.shape, (2, m.num_experts))
        torch.testing.assert_close(gates, expected)

    def test_compute_loss(self):
        loss_options = [0.0, 0.1]
        for cross_option in loss_options:
            for coeff_option in loss_options:
                for zero_option in loss_options:
                    for mutual_option in loss_options:
                        message = f"Running test with cross_option={cross_option}, coeff_option={coeff_option}, zero_option={zero_option}, mutual_option={mutual_option}"
                        with self.subTest(msg=message):
                            cfg = self.preset(
                                top_k=3,
                                noisy_topk_flag=True,
                                threshold=0.8,
                                filter_above_threshold=True,
                                normalize_probabilities_flag=True,
                                coefficient_of_variation_loss_weight=coeff_option,
                                switch_loss_weight=cross_option,
                                zero_centred_loss_weight=zero_option,
                                mutual_information_loss_weight=mutual_option,
                            )
                            m = SamplerTopk(cfg)

                            batch_size = 3
                            sequence_length = 4
                            shape = (batch_size * sequence_length, m.num_experts)

                            logits = torch.randn(*shape)
                            full_probabilities = torch.softmax(
                                torch.randn(*shape), dim=-1
                            )
                            sampled_probabilities = torch.softmax(
                                torch.randn(batch_size * sequence_length, m.top_k),
                                dim=-1,
                            )
                            indices = torch.randint(
                                0,
                                m.num_experts,
                                (batch_size * sequence_length, m.top_k),
                            )
                            mask = torch.ones(batch_size, sequence_length).reshape(
                                -1, 1
                            )
                            unmasked_token = 0
                            mask[unmasked_token, :] = 1

                            total_loss = m._compute_loss(
                                logits,
                                full_probabilities,
                                sampled_probabilities,
                                indices,
                                mask,
                            )

                            if (
                                coeff_option == 0.0
                                and cross_option == 0.0
                                and zero_option == 0.0
                                and mutual_option == 0.0
                            ):
                                self.assertTrue(total_loss == 0)
                            else:
                                self.assertTrue(total_loss > 0)
                            self.assertIsInstance(total_loss, torch.Tensor)


class TestSamplerFull(SamplerTestCase):
    def test_sample_probabilities_and_indices(self):
        cfg = self.preset(
            top_k=10,
            num_experts=10,
        )
        m = SamplerFull(cfg)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        probability, indices = m._sample_probabilities_and_indices(probabilities)
        self.assertEqual(probability.shape, (batch_size, m.num_experts))
        self.assertEqual(indices, None)

    def test_get_probabilities_and_indices(self):
        noisy_flag_options = [True, False]
        threshold_options = [0.0, 0.2]
        filter_above_threshold_options = [True, False]
        for noisy_flag in noisy_flag_options:
            for thresh in threshold_options:
                for filter_flag in filter_above_threshold_options:
                    message = (
                        f"noisy_flag={noisy_flag}, "
                        f"threshold={thresh}, "
                        f"filter_flag={filter_flag}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            top_k=10,
                            num_experts=10,
                            noisy_topk_flag=noisy_flag,
                            threshold=thresh,
                            filter_above_threshold=filter_flag,
                            coefficient_of_variation_loss_weight=0.0,
                            switch_loss_weight=0.0,
                            zero_centred_loss_weight=0.0,
                            mutual_information_loss_weight=0.0,
                        )
                        m = SamplerFull(cfg)

                        batch_size = 3
                        sequence_length = 4
                        num_expert_dim = m.num_experts
                        if noisy_flag:
                            num_expert_dim = m.num_experts * 2
                        shape = (
                            batch_size * sequence_length,
                            num_expert_dim,
                        )
                        logits = torch.softmax(torch.randn(*shape), dim=-1)
                        mask = torch.ones(batch_size, sequence_length).reshape(
                            -1, 1
                        )
                        probabilities, selected_indices, skip_mask, loss = (
                            m.get_probabilities_and_indices(logits, mask)
                        )
                        self.assertEqual(
                            probabilities.shape,
                            (batch_size * sequence_length, m.top_k),
                        )
                        self.assertIsNone(selected_indices)
                        self.assertTrue(loss == 0)

    def test__apply_dynamic_topk_threshold_mask(self):
        threshold_options = [0.0, 0.1]

        for threshold_option in threshold_options:
            message = f"threshold={threshold_option}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    top_k=10,
                    num_experts=10,
                    threshold=threshold_option,
                )
                m = SamplerFull(cfg)

                batch_size = 3
                sequence_length = 4
                shape = (batch_size * sequence_length, m.num_experts)
                probabilities = torch.softmax(torch.randn(*shape), dim=-1)

                masked_probabilities = (
                    m._SamplerFull__apply_dynamic_topk_threshold_mask(probabilities)
                )

                if threshold_option > 0.0:
                    self.assertEqual(masked_probabilities.shape, shape)
                    self.assertFalse(
                        torch.allclose(
                            probabilities.round(decimals=4),
                            masked_probabilities.round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )
                else:
                    self.assertEqual(masked_probabilities.shape, shape)
                    self.assertTrue(
                        torch.allclose(
                            probabilities.round(decimals=4),
                            masked_probabilities.round(decimals=4),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )

    def test_apply_dynamic_topk_threshold_mask_normalizes_unmasked_values(self):
        cfg = self.preset(
            top_k=4,
            num_experts=4,
            threshold=0.25,
            normalize_probabilities_flag=True,
        )
        m = SamplerFull(cfg)
        probabilities = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.1, 0.3, 0.1],
            ]
        )

        masked_probabilities = m._SamplerFull__apply_dynamic_topk_threshold_mask(
            probabilities
        )

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.3 / 0.7, 0.4 / 0.7],
                [0.5 / 0.8, 0.0, 0.3 / 0.8, 0.0],
            ]
        )
        torch.testing.assert_close(masked_probabilities, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            masked_probabilities.sum(dim=-1),
            torch.ones(2),
            atol=1e-5,
            rtol=1e-5,
        )


class TestSamplerModel(SamplerTestCase):
    def test_init(self):
        cfg = self.preset()
        model = SamplerModel(cfg)
        self.assertTrue(isinstance(model.sampler_model, SamplerTopk))

    def test_model_type_storage(self):
        topk_options = [1, 3, 5]
        model_types = [SamplerSparse, SamplerTopk, SamplerFull]

        for top_k, model_type in zip(topk_options, model_types):
            message = f"Testing configuration with top_k={top_k}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    top_k=top_k,
                    num_experts=5,
                )
                model = SamplerModel(cfg)
                self.assertIsInstance(model.sampler_model, model_type)

    def test_sample_probs_and_indexes_logits_only(self):
        topk_options = [1, 3, 5]
        model_types = [SamplerSparse, SamplerTopk, SamplerFull]

        for top_k, model_type in zip(topk_options, model_types):
            message = f"Running test with configuration top_k={top_k}, model_type={model_type}"
            with self.subTest(msg=message):
                num_experts = 5
                loss = 0.0 if num_experts == top_k else 0.5
                cfg = self.preset(
                    top_k=top_k,
                    num_experts=num_experts,
                    coefficient_of_variation_loss_weight=loss,
                    switch_loss_weight=loss,
                    zero_centred_loss_weight=loss,
                )
                m = SamplerModel(cfg)

                batch_size = 3
                sequence_length = 4
                logits = torch.randn(batch_size * sequence_length, num_experts)

                probabilities, indices, skip_mask, loss = (
                    m.sample_probabilities_and_indices(logits)
                )

                if top_k == 1:
                    expected_output_shape = (batch_size * sequence_length,)
                else:
                    expected_output_shape = (batch_size * sequence_length, top_k)

                self.assertEqual(probabilities.shape, expected_output_shape)
                if top_k == num_experts:
                    self.assertIsNone(indices)
                else:
                    self.assertEqual(indices.shape, expected_output_shape)
                self.assertIsNone(skip_mask)
                if num_experts == top_k:
                    self.assertTrue(loss == 0.0)
                else:
                    self.assertTrue(loss > 0.0)

                self.assertIsInstance(m.sampler_model, model_type)

    def test_sample_probs_and_indexes_applies_skip_mask(self):
        cfg = self.preset(top_k=3, num_experts=5, threshold=0.2)
        model = SamplerModel(cfg)
        logits = torch.tensor(
            [
                [5.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 4.0, 0.0],
            ]
        )
        skip_mask = torch.tensor([[1.0], [0.0], [1.0]])

        probabilities, indices, updated_skip_mask, loss = (
            model.sample_probabilities_and_indices(logits, skip_mask)
        )

        self.assertEqual(probabilities.shape, (3, 3))
        self.assertEqual(indices.shape, (3, 3))
        torch.testing.assert_close(probabilities[1], torch.zeros(3))
        torch.testing.assert_close(updated_skip_mask, skip_mask)
        self.assertIsInstance(loss, torch.Tensor)

    def test_sample_probs_and_indexes_logits_only_and_skip_mask(self):
        topk_options = [1, 3, 5]
        model_types = [SamplerSparse, SamplerTopk, SamplerFull]

        for top_k, model_type in zip(topk_options, model_types):
            message = f"Running test with configuration top_k={top_k}, model_type={model_type}"
            with self.subTest(msg=message):
                num_experts = 5
                loss = 0.0 if num_experts == top_k else 0.5
                cfg = self.preset(
                    top_k=top_k,
                    num_experts=num_experts,
                    coefficient_of_variation_loss_weight=loss,
                    switch_loss_weight=loss,
                    zero_centred_loss_weight=loss,
                )
                m = SamplerModel(cfg)

                batch_size = 3
                sequence_length = 4
                logits = torch.randn(batch_size * sequence_length, num_experts)
                mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
                unmasked_token = 0
                mask[unmasked_token + 1 :, :] = 0

                probabilities, indices, skip_mask, loss = (
                    m.sample_probabilities_and_indices(logits, mask)
                )

                if top_k == 1:
                    expected_output_shape = (batch_size * sequence_length,)
                else:
                    expected_output_shape = (batch_size * sequence_length, top_k)

                self.assertEqual(probabilities.shape, expected_output_shape)
                self.assertEqual(skip_mask.shape, (batch_size * sequence_length, 1))
                self.assertIsInstance(m.sampler_model, model_type)

                if top_k == num_experts:
                    self.assertIsNone(indices)
                else:
                    self.assertEqual(indices.shape, expected_output_shape)
                if num_experts == top_k:
                    self.assertTrue(loss == 0.0)
                else:
                    self.assertTrue(loss > 0.0)
