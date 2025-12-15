import torch
import unittest
from math import prod

from Emperor.config import ModelConfig
from Emperor.sampler.model import SamplerModel
from Emperor.sampler.utils.presets import SamplerPresets
from Emperor.sampler.utils.samplers import (
    SamplerBase,
    SamplerFull,
    SamplerSparse,
    SamplerTopk,
)


class TestProbabilitySampler(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = SamplerPresets.sampler_preset() if config is None else config

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_init(self):
        cfg = SamplerPresets.sampler_preset()
        c = cfg.sampler_model_config
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
                cfg = SamplerPresets.sampler_preset(
                    normalize_probabilities_flag=flag,
                )
                m = SamplerBase(cfg)
                probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
                result = m._normalize_probabilities(probs)

                if flag:
                    distribution_check = result.sum(dim=-1)
                    expected_distribution = torch.tensor([1.0, 1.0])
                    self.assertTrue(
                        torch.allclose(distribution_check, expected_distribution)
                    )
                else:
                    self.assertTrue(torch.allclose(probs, result))

    def test__probability_sampling_strategy(self):
        cfg = SamplerPresets.sampler_preset()
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
                    cfg = SamplerPresets.sampler_preset(
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
                            self.assertTrue(torch.allclose(output, expected_mask))
                        else:
                            expected_mask[1] = 0.0
                            self.assertTrue(torch.allclose(output, expected_mask))
                    else:
                        self.assertTrue(torch.allclose(skip_mask, output))

    def test__apply_skip_mask__threshold__zero(self):
        threshold_options = [0.0, 0.4]

        for threshold_option in threshold_options:
            message = f"Testing configuration with threshold={threshold_option}, "
            with self.subTest(msg=message):
                cfg = SamplerPresets.sampler_preset(
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
                    self.assertTrue(torch.allclose(masked_probs, probs))
                    self.assertTrue(torch.allclose(router_logit_scores, logits))
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
                cfg = SamplerPresets.sampler_preset(
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
                    self.assertTrue(torch.allclose(logits, result))

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
                    cfg = SamplerPresets.sampler_preset(
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
                                torch.sum(masked_probabilities[0, :], dim=-1),
                                torch.tensor(1.0).float(),
                            )
                        )
                    elif noisy_topk_flag_option and (threshold_option == 0.0):
                        self.assertTrue(
                            masked_probabilities.shape, (batch_size, num_expert_dim)
                        )
                    else:
                        self.assertTrue(
                            masked_probabilities.shape,
                            router_logit_scores.shape,
                        )


class TestSamplerSparse(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = SamplerPresets.sampler_preset() if config is None else config

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_sample_probabilities_and_indices(self):
        cfg = SamplerPresets.sampler_preset()
        m = SamplerSparse(cfg)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        probability, indices = m._sample_probabilities_and_indices(probabilities)
        self.assertEqual(probability.shape, (batch_size,))
        self.assertEqual(indices.shape, (batch_size,))

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
                                    with self.assertRaises(AssertionError):
                                        cfg = SamplerPresets.sampler_preset(
                                            normalize_probabilities_flag=normalize_flag,
                                        )
                                        m = SamplerSparse(cfg)
                                else:
                                    cfg = SamplerPresets.sampler_preset(
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
                                    shape = (
                                        batch_size * sequence_length,
                                        m.num_experts,
                                    )
                                    logits = torch.softmax(torch.randn(*shape), dim=-1)
                                    mask = torch.ones(
                                        batch_size, sequence_length
                                    ).reshape(-1, 1)
                                    probabilities, selected_indices, skip_mask, loss = (
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
                                    if loss > 0:
                                        self.assertTrue(loss >= 0)

    def test__prepare_loss_skip_mask(self):
        input_options = [None, torch.ones(3, 4)]
        for input_mask in input_options:
            message = f"Testing configuration with skip_mask={input_mask}"
            with self.subTest(msg=message):
                cfg = SamplerPresets.sampler_preset()
                m = SamplerSparse(cfg)

                skip_maks = m._SamplerSparse__prepare_loss_skip_mask(input_mask)

                if input_mask is None:
                    self.assertIsNone(skip_maks)
                else:
                    self.assertTrue(
                        torch.allclose(skip_maks, input_mask.reshape(-1, 1))
                    )

    def test__prepare_loss_gates(self):
        cfg = SamplerPresets.sampler_preset(top_k=1)
        m = SamplerSparse(cfg)

        batch_size = 3

        shape = (batch_size, m.top_k)
        sampled_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        indices = torch.randint(0, m.num_experts, (batch_size, m.top_k))

        gates = m._SamplerSparse__prepare_loss_gates(sampled_probabilities, indices)

        self.assertEqual(gates.shape, (batch_size, m.num_experts))
        self.assertTrue(
            torch.allclose(torch.sum(gates, dim=-1), torch.ones(batch_size).float())
        )

    def test_compute_loss(self):
        loss_options = [0.0, 0.1]
        for cross_option in loss_options:
            for coeff_option in loss_options:
                for zero_option in loss_options:
                    message = f"Running test with cross_option={cross_option}, coeff_option={coeff_option}, zero_option={zero_option}"
                    with self.subTest(msg=message):
                        cfg = SamplerPresets.sampler_preset(
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
                        sampled_probabilities = torch.softmax(
                            torch.randn(*shape), dim=-1
                        )
                        indices = torch.randint(0, m.num_experts, (batch_size, m.top_k))
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


class TestSamplerTopk(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = SamplerPresets.sampler_preset() if config is None else config

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_sample_probabilities_and_indices(self):
        cfg = SamplerPresets.sampler_preset()
        m = SamplerTopk(cfg)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        probability, indices = m._sample_probabilities_and_indices(probabilities)
        self.assertEqual(probability.shape, (batch_size, m.top_k))
        self.assertEqual(indices.shape, (batch_size, m.top_k))

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
                                cfg = SamplerPresets.sampler_preset(
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
                                probabilities, selected_indices, skip_mask, loss = (
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
                                if loss > 0:
                                    self.assertTrue(loss >= 0)

    def test__prepare_loss_skip_mask(self):
        input_options = [None, torch.ones(3, 4)]
        for input_mask in input_options:
            message = f"Testing configuration with skip_mask={input_mask}"
            with self.subTest(msg=message):
                cfg = SamplerPresets.sampler_preset()
                m = SamplerTopk(cfg)

                skip_maks = m._SamplerTopk__prepare_loss_skip_mask(input_mask)
                if input_mask is None:
                    self.assertIsNone(skip_maks)
                else:
                    self.assertTrue(
                        torch.allclose(skip_maks, input_mask.reshape(-1, 1))
                    )

    def test__prepare_loss_gates(self):
        cfg = SamplerPresets.sampler_preset(top_k=1)
        m = SamplerTopk(cfg)

        batch_size = 3
        shape = (batch_size, m.top_k)
        sampled_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        indices = torch.randint(0, m.num_experts, (batch_size, m.top_k))

        gates = m._SamplerTopk__prepare_loss_gates(sampled_probabilities, indices)

        self.assertEqual(gates.shape, (batch_size, m.num_experts))
        self.assertTrue(
            torch.allclose(torch.sum(gates, dim=-1), torch.ones(batch_size).float())
        )

    def test_compute_loss(self):
        loss_options = [0.0, 0.1]
        for cross_option in loss_options:
            for coeff_option in loss_options:
                for zero_option in loss_options:
                    for mutual_option in loss_options:
                        message = f"Running test with cross_option={cross_option}, coeff_option={coeff_option}, zero_option={zero_option}, mutual_option={mutual_option}"
                        with self.subTest(msg=message):
                            cfg = SamplerPresets.sampler_preset(
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
                                torch.randn(batch_size, m.top_k), dim=-1
                            )
                            indices = torch.randint(
                                0, m.num_experts, (batch_size, m.top_k)
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


class TestSamplerFull(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = SamplerPresets.sampler_preset() if config is None else config

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_sample_probabilities_and_indices(self):
        cfg = SamplerPresets.sampler_preset(
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
        normalize_probabilities_flag = [True, False]

        for noisy_flag in noisy_flag_options:
            for thresh in threshold_options:
                for filter_flag in filter_above_threshold_options:
                    for normalize_flag in normalize_probabilities_flag:
                        message = (
                            f"noisy_flag={noisy_flag}, "
                            f"threshold={thresh}, "
                            f"filter_flag={filter_flag}, "
                            f"normalize_flag={normalize_flag}"
                        )
                        with self.subTest(msg=message):
                            cfg = SamplerPresets.sampler_preset(
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
                cfg = SamplerPresets.sampler_preset(
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
                        torch.allclose(probabilities, masked_probabilities)
                    )
                else:
                    self.assertEqual(masked_probabilities.shape, shape)
                    self.assertTrue(torch.allclose(probabilities, masked_probabilities))


class TestSamplerModel(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = SamplerPresets.sampler_preset() if config is None else config

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_init(self):
        cfg = SamplerPresets.sampler_preset()
        model = SamplerModel(cfg)
        self.assertTrue(isinstance(model.sampler_model, SamplerTopk))

    def test_model_type_storage(self):
        topk_options = [1, 3, 5]
        model_types = [SamplerSparse, SamplerTopk, SamplerFull]

        for top_k, model_type in zip(topk_options, model_types):
            message = f"Testing configuration with top_k={top_k}"
            with self.subTest(msg=message):
                cfg = SamplerPresets.sampler_preset(
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
                cfg = SamplerPresets.sampler_preset(
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

    def test_sample_probs_and_indexes_logits_only_and_skip_mask(self):
        topk_options = [1, 3, 5]
        model_types = [SamplerSparse, SamplerTopk, SamplerFull]

        for top_k, model_type in zip(topk_options, model_types):
            message = f"Running test with configuration top_k={top_k}, model_type={model_type}"
            with self.subTest(msg=message):
                num_experts = 5
                loss = 0.0 if num_experts == top_k else 0.5
                cfg = SamplerPresets.sampler_preset(
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
                mask[unmasked_token, :] = 1

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
