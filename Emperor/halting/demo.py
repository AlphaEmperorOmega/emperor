import torch

from Emperor.halting.config import HaltingConfig
from Emperor.halting.options import HaltingOptions
from Emperor.halting.options.stick_breaking import StickBreaking

# ── Configuration ─────────────────────────────────────────────────────────────

NUM_STEPS  = 6
BATCH_SIZE = 1
SEQ_LEN    = 1
HIDDEN_DIM = 4
THRESHOLD  = 0.5

# ── Setup ─────────────────────────────────────────────────────────────────────

cfg = HaltingConfig(
    halting_option=HaltingOptions.STICK_BREAKING,
    input_dim=HIDDEN_DIM,
    threshold=THRESHOLD,
    halting_dropout=0.0,
)

model = StickBreaking(cfg)
model.eval()

hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

# ── Loop ──────────────────────────────────────────────────────────────────────

print(f"\nStick-Breaking  |  steps: {NUM_STEPS}  |  threshold: {THRESHOLD}\n")
print(f"{'Step':<6} {'P(halt this step)':>20} {'P(survival)':>14} {'Accumulated':>14} {'Stick left':>12} {'Status':>10}")
print("-" * 80)

state        = None
prev_accumulated = 0.0

for loop in range(NUM_STEPS):
    state = model.forward(state, hidden)

    halting_prob = state.accumulated_halt_prob.item()
    survival     = state.log_never_halt.exp().item()
    stick_left   = 1.0 - halting_prob
    halted       = state.halt_mask.any().item()
    p_this_step  = halting_prob - prev_accumulated

    label  = "init" if loop == 0 else f"step {state.step}"
    status = "HALTED" if halted else "active"

    print(
        f"{label:<6} {p_this_step:>20.4f} {survival:>14.4f} "
        f"{halting_prob:>14.4f} {stick_left:>12.4f} {status:>10}"
    )

    prev_accumulated = halting_prob

    if state.halt_mask.all():
        print(f"\n  All tokens halted — stopping early.\n")
        break

print("-" * 80)

assert state is not None

# ── Halt gating ───────────────────────────────────────────────────────────────

soft_halted_hidden, expected_step = model.halt_gating(state, hidden)

print(f"\n  Expected computation depth : {expected_step.item():.4f}")
print(f"  Soft-halted hidden         : {soft_halted_hidden.squeeze().tolist()}\n")
