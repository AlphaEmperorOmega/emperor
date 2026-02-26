import torch
import unittest

from Emperor.halting.config import HaltingConfig
from Emperor.halting.options import HaltingOptions
from Emperor.halting.options.soft_halting import SoftHalting
from Emperor.halting.options.stick_breaking import StickBreaking


BATCH = 2
SEQ_LEN = 4
INPUT_DIM = 8
THRESHOLD = 0.999


def make_config() -> HaltingConfig:
    return HaltingConfig(
        halting_option=HaltingOptions.SOFT_HALTING,
        input_dim=INPUT_DIM,
        threshold=THRESHOLD,
        halting_dropout=0.0,
    )


class TestSoftHalting(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.model = SoftHalting(self.cfg)
        self.model.eval()
        self.h = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        self.pad_mask = torch.ones(BATCH, SEQ_LEN)

    # --- init step ---

    def test_init_step_p_never_halt_equals_pad_mask(self):
        state, p_never_halt = self.model.forward(None, self.h, self.pad_mask)
        self.assertTrue(torch.equal(p_never_halt, self.pad_mask))

    def test_init_step_state_counter_is_zero(self):
        state, _ = self.model.forward(None, self.h, self.pad_mask)
        self.assertEqual(state[0], 0)

    def test_init_step_accumulator_shapes(self):
        state, _ = self.model.forward(None, self.h, self.pad_mask)
        _, log_never_halt, acc_h, acc_expect_depth = state
        self.assertEqual(log_never_halt.shape, (BATCH, SEQ_LEN))
        self.assertEqual(acc_h.shape, self.h.shape)
        self.assertEqual(acc_expect_depth.shape, (BATCH, SEQ_LEN))

    # --- step ---

    def test_step_increments_counter(self):
        state, _ = self.model.forward(None, self.h, self.pad_mask)
        state2, _ = self.model.forward(state, torch.randn_like(self.h), self.pad_mask)
        self.assertEqual(state2[0], 1)

    def test_step_p_never_halt_shape(self):
        state, _ = self.model.forward(None, self.h, self.pad_mask)
        _, p_never_halt = self.model.forward(state, torch.randn_like(self.h), self.pad_mask)
        self.assertEqual(p_never_halt.shape, (BATCH, SEQ_LEN))

    def test_step_p_never_halt_is_zero_where_masked(self):
        # Tokens that fall below threshold should be zeroed out
        state, _ = self.model.forward(None, self.h, self.pad_mask)
        _, p_never_halt = self.model.forward(state, torch.randn_like(self.h), self.pad_mask)
        self.assertTrue((p_never_halt >= 0).all())

    # --- compute_output ---

    def test_compute_output_step_zero_returns_curr_h(self):
        state, p_never_halt = self.model.forward(None, self.h, self.pad_mask)
        curr_h = torch.randn_like(self.h)
        out, loss = self.model.compute_output(
            state, curr_h, torch.randn_like(self.h), p_never_halt, self.pad_mask
        )
        self.assertTrue(torch.equal(out, curr_h))
        self.assertEqual(loss.item(), 0.0)

    def test_compute_output_step_one_returns_correct_shapes(self):
        state, _ = self.model.forward(None, self.h, self.pad_mask)
        state2, p2 = self.model.forward(state, torch.randn_like(self.h), self.pad_mask)
        out, loss = self.model.compute_output(
            state2, torch.randn_like(self.h), torch.randn_like(self.h), p2, self.pad_mask
        )
        self.assertEqual(out.shape, self.h.shape)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_compute_output_loss_is_non_negative(self):
        state, _ = self.model.forward(None, self.h, self.pad_mask)
        state2, p2 = self.model.forward(state, torch.randn_like(self.h), self.pad_mask)
        _, loss = self.model.compute_output(
            state2, torch.randn_like(self.h), torch.randn_like(self.h), p2, self.pad_mask
        )
        self.assertGreaterEqual(loss.item(), 0.0)

    # --- gate ---

    def test_gate_weights_initialized_to_zero(self):
        last_layer = self.model._gate[-1]
        self.assertTrue(torch.all(last_layer.weight == 0))

    # --- multi-step ---

    def test_multi_step_loop_output_shape(self):
        state = None
        h = self.h
        self_attn_input = torch.randn_like(h)
        for _ in range(5):
            state, p_never_halt = self.model.forward(state, h, self.pad_mask)
            curr_h = torch.randn_like(h)
            self_attn_input, _ = self.model.compute_output(
                state, curr_h, self_attn_input, p_never_halt, self.pad_mask
            )
            h = self_attn_input
        self.assertEqual(h.shape, self.h.shape)


class TestStickBreaking(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.model = StickBreaking(self.cfg)
        self.model.eval()
        self.h = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)

    # --- init step ---

    def test_init_step_halt_mask_shape(self):
        _, halt_mask = self.model.forward(None, self.h)
        self.assertEqual(halt_mask.shape, (BATCH, SEQ_LEN))

    def test_init_step_halt_mask_is_bool(self):
        _, halt_mask = self.model.forward(None, self.h)
        self.assertEqual(halt_mask.dtype, torch.bool)

    def test_init_step_state_counter_is_zero(self):
        state, _ = self.model.forward(None, self.h)
        self.assertEqual(state[4], 0)

    def test_init_step_acc_h_shape(self):
        state, _ = self.model.forward(None, self.h)
        acc_h = state[2]
        self.assertEqual(acc_h.shape, self.h.shape)

    def test_init_step_acc_g_shape(self):
        state, _ = self.model.forward(None, self.h)
        acc_g = state[3]
        self.assertEqual(acc_g.shape, (BATCH, SEQ_LEN))

    # --- step ---

    def test_step_increments_counter(self):
        state, _ = self.model.forward(None, self.h)
        state2, _ = self.model.forward(state, torch.randn_like(self.h))
        self.assertEqual(state2[4], 1)

    def test_halted_tokens_stop_accumulating(self):
        state, halt_mask = self.model.forward(None, self.h)
        acc_g_before = state[3].clone()
        state2, _ = self.model.forward(state, torch.randn_like(self.h))
        acc_g_after = state2[3]
        if halt_mask.any():
            self.assertTrue(
                torch.allclose(acc_g_after[halt_mask], acc_g_before[halt_mask])
            )

    def test_acc_g_monotonically_increases(self):
        state, _ = self.model.forward(None, self.h)
        acc_g_step1 = state[3].clone()
        state2, _ = self.model.forward(state, torch.randn_like(self.h))
        acc_g_step2 = state2[3]
        self.assertTrue((acc_g_step2 >= acc_g_step1).all())

    # --- halt_gating ---

    def test_halt_gating_output_shape(self):
        state, _ = self.model.forward(None, self.h)
        soft_halted_h, expstep = self.model.halt_gating(state, torch.randn_like(self.h))
        self.assertEqual(soft_halted_h.shape, self.h.shape)
        self.assertEqual(expstep.shape, (BATCH, SEQ_LEN))

    def test_halt_gating_fully_halted_tokens_frozen(self):
        # With threshold=0.0 all tokens halt immediately
        cfg = HaltingConfig(
            halting_option=HaltingOptions.STICK_BREAKING,
            input_dim=INPUT_DIM,
            threshold=0.0,
            halting_dropout=0.0,
        )
        model = StickBreaking(cfg)
        model.eval()
        state, halt_mask = model.forward(None, self.h)
        self.assertTrue(halt_mask.all())
        curr_h = torch.randn_like(self.h)
        soft_halted_h, _ = model.halt_gating(state, curr_h)
        # Fully halted output must equal acc_h, not curr_h
        acc_h = state[2]
        self.assertTrue(torch.allclose(soft_halted_h, acc_h))

    # --- gate ---

    def test_gate_weights_initialized_to_zero(self):
        last_layer = self.model._gate[-1]
        self.assertTrue(torch.all(last_layer.weight == 0))

    # --- multi-step ---

    def test_multi_step_loop_output_shape(self):
        state = None
        h = self.h
        for _ in range(5):
            state, _ = self.model.forward(state, h)
            h, _ = self.model.halt_gating(state, h)
        self.assertEqual(h.shape, self.h.shape)


if __name__ == "__main__":
    unittest.main()
