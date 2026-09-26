import math

import torch
from torch import nn
from torchmetrics import Metric

from emperor.decoding.hierarchical import HierarchicalByteDecoderOutput

from ._records import HierarchicalLanguageModelStepOutput


class _AccumulatedRatio(Metric):
    """Sum both sides before division, including control-only document batches."""

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "numerator",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "denominator",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, numerator, denominator):
        self.numerator += numerator.detach().double()
        self.denominator += denominator

    def compute(self):
        ratio = self.numerator / self.denominator.clamp_min(1)
        return torch.where(
            self.denominator > 0, ratio, ratio.new_full((), float("nan"))
        )


class HierarchicalLanguageModelHandler(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics = nn.ModuleDict(
            {
                f"{stage}_stage": nn.ModuleDict(
                    {
                        name: _AccumulatedRatio()
                        for name in (
                            "loss",
                            "symbol_cross_entropy",
                            "auxiliary_loss",
                            "bits_per_byte",
                        )
                    }
                )
                for stage in ("train", "validation", "test")
            }
        )

    def step(self, model, batch):
        batch = batch.to(model.device)
        output = model(batch)
        if not isinstance(output, HierarchicalByteDecoderOutput):
            raise TypeError(
                "hierarchical batches require HierarchicalByteDecoderOutput"
            )
        if output.logits.shape != (batch.labels.numel(), 258):
            raise ValueError(
                "packed hierarchical logits must align with all byte/control labels"
            )
        if (
            batch.labels.dtype != torch.long
            or batch.labels.ndim != 1
            or bool(((batch.labels < 0) | (batch.labels >= 258)).any())
        ):
            raise ValueError(
                "packed labels must be a rank-1 long tensor of byte/control IDs"
            )
        expected_offsets = torch.cat(
            (
                batch.byte_lengths.new_zeros(1),
                torch.where(batch.attention_mask, batch.byte_lengths + 1, 0)
                .flatten()
                .cumsum(0),
            )
        )
        if not torch.equal(output.token_offsets, expected_offsets):
            raise ValueError(
                "decoder offsets must match the batch's valid prediction positions"
            )
        lengths = batch.byte_lengths.flatten()
        valid = batch.attention_mask.flatten()
        prefixes = batch.byte_prefix_ids.reshape(
            lengths.numel(), batch.byte_prefix_ids.shape[-1]
        )
        byte_positions = torch.arange(prefixes.shape[-1], device=prefixes.device)
        byte_mask = valid[:, None] & (byte_positions < lengths[:, None])
        packed_positions = expected_offsets[:-1, None] + byte_positions
        if not torch.equal(
            batch.labels[packed_positions[byte_mask]], prefixes[byte_mask]
        ):
            raise ValueError(
                "packed labels must align with the teacher-forced byte prefixes"
            )
        controls = batch.labels[expected_offsets[1:][valid] - 1]
        if bool(
            ((controls != 256) & ((controls != 257) | (lengths[valid] != 0))).any()
        ):
            raise ValueError(
                "labels must end each token with EOW, or encode EOS as a single prediction"
            )
        if type(batch.byte_count) is not int or batch.byte_count != int(
            batch.byte_lengths[batch.attention_mask].sum()
        ):
            raise ValueError("byte_count must equal the number of target bytes")
        if batch.labels.numel():
            nll = nn.functional.cross_entropy(
                output.logits, batch.labels, reduction="sum"
            )
            # Normalize the symbol objective before adding native auxiliary losses.
            # This is not a word-perplexity objective or the paper's written sum.
            cross_entropy = nll / batch.labels.numel()
        else:
            # Retain a differentiable zero for an entirely masked training batch.
            nll = next(model.parameters()).sum() * 0
            cross_entropy = nll
        auxiliary_loss = model._auxiliary_loss.resolve(
            output.loss, reference=output.logits
        )
        return HierarchicalLanguageModelStepOutput(
            total_loss=cross_entropy + auxiliary_loss,
            cross_entropy=cross_entropy,
            logits=output.logits,
            labels=batch.labels,
            auxiliary_loss=auxiliary_loss,
            nll_sum=nll,
            byte_count=batch.byte_count,
        )

    def log(self, log_fn, stage, output):
        count = output.labels.numel()
        metrics = self.metrics[f"{stage}_stage"]
        metrics["loss"].update(output.total_loss * count, count)
        metrics["symbol_cross_entropy"].update(output.nll_sum, count)
        metrics["auxiliary_loss"].update(output.auxiliary_loss * count, count)
        metrics["bits_per_byte"].update(output.nll_sum / math.log(2), output.byte_count)
        log_fn(
            {f"{stage}/{name}": metric for name, metric in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "test",
            batch_size=1,
        )
