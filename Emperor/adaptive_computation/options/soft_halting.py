class ACTWrapper(nn.Module):
    def __init__(self, mod, threshold=ACT_THRESHOLD, halting_dropout=0):
        super(ACTWrapper, self).__init__()
        self._gate = nn.Sequential(
            nn.Linear(mod.embed_dim, mod.embed_dim),
            nn.GELU(),
            nn.Dropout(halting_dropout),
            nn.Linear(mod.embed_dim, 2, bias=False),
        )
        nn.init.zeros_(self._gate[-1].weight)
        self.threshold = threshold
        self.mod = mod

    def gate(self, h):
        logits = self._gate(h)
        return F.log_softmax(logits, dim=-1)

    def update_halting(self, log_g, log_never_halt):
        log_halt = log_never_halt[..., None] + log_g
        log_never_halt = log_halt[..., 0]
        p = torch.exp(log_halt[..., 1])
        return p, log_never_halt

    def forward(
        self,
        prev_act_state,
        prev_h,
        self_attn_input,
        pad_mask,
        layer_idx,
        *args,
        **kwargs,
    ):
        if prev_act_state is None:
            log_never_halt = acc_expect_depth = torch.zeros_like(prev_h[..., 0])
            acc_h = torch.zeros_like(prev_h)
            i = 0
            p_never_halt = pad_mask
        else:
            (i, log_never_halt, acc_h, acc_expect_depth) = prev_act_state
            log_g = self.gate(prev_h)
            p, log_never_halt = self.update_halting(log_g, log_never_halt)
            acc_h = acc_h + p[..., None] * prev_h
            acc_expect_depth = acc_expect_depth + i * p
            p_never_halt = log_never_halt.exp()
            p_never_halt = (
                p_never_halt.masked_fill((p_never_halt < (1 - self.threshold)), 0)
                * pad_mask
            )
            p_never_halt = p_never_halt.contiguous()
            i = i + 1

        curr_act_state = (i, log_never_halt, acc_h, acc_expect_depth)
        outputs = self.mod(
            prev_h, self_attn_input, p_never_halt, layer_idx, *args, **kwargs
        )

        curr_h = outputs[0]
        if prev_act_state is not None:
            self_attn_input = torch.where(
                p_never_halt[..., None] < (1 - self.threshold),
                self_attn_input,
                (acc_h + p_never_halt[..., None] * curr_h).type_as(self_attn_input),
            )
            act_loss = (acc_expect_depth + p_never_halt * i) * pad_mask
            act_loss = act_loss.sum() / pad_mask.sum()
        else:
            self_attn_input = curr_h
            act_loss = 0

        return curr_act_state, outputs, self_attn_input, act_loss
