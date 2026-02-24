class StickBreakingACT(nn.Module):
    def __init__(self, hidden_size, threshold=0.999):
        super(StickBreakingACT, self).__init__()
        self._gate = nn.Sequential(nn.Linear(hidden_size, 2, bias=False))
        nn.init.zeros_(self._gate[-1].weight)
        self.threshold = threshold

    def gate(self, h):
        logits = self._gate(h)
        if self.training:
            logits = logits + torch.randn_like(logits)
        return F.log_softmax(logits, dim=-1)

    def update_halting(self, curr_log_g, prev_out, prev_state):
        if prev_state is None:
            step = 0
            curr_log_halt = curr_log_g
            g = torch.exp(curr_log_halt[..., 1])
            curr_acc_g = g
            curr_acc_h = g[..., None] * prev_out
            curr_expstep = 0.0
        else:
            (
                prev_halt_mask,
                prev_log_never_halt,
                prev_acc_h,
                prev_acc_g,
                prev_step,
                prev_acc_expstep,
            ) = prev_state
            step = prev_step + 1
            curr_log_halt = prev_log_never_halt[..., None] + curr_log_g
            g = torch.exp(curr_log_halt[..., 1])
            g = g.masked_fill(prev_halt_mask, 0.0)
            curr_acc_g = prev_acc_g + g
            curr_acc_h = prev_acc_h + g[..., None] * prev_out
            curr_expstep = prev_acc_expstep + g * step

        halt_mask = curr_acc_g >= self.threshold
        curr_state = (
            halt_mask,
            curr_log_halt[..., 0],
            curr_acc_h,
            curr_acc_g,
            step,
            curr_expstep,
        )

        return curr_state, halt_mask

    def forward(self, prev_state, prev_out):
        log_g = self.gate(prev_out)
        return self.update_halting(log_g, prev_out, prev_state)

    def halt_gating(self, curr_state, curr_h):
        halt_mask, _, curr_acc_h, curr_acc_g, step, curr_expstep = curr_state
        soft_halted_h = curr_acc_h + (1 - curr_acc_g)[..., None] * curr_h
        expstep = curr_expstep + (step + 1) * (1 - curr_acc_g)
        if halt_mask.any():
            soft_halted_h.masked_scatter_(halt_mask[..., None], curr_acc_h[halt_mask])
        return soft_halted_h, expstep
