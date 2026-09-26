"""Bounded UTF-8 generation over a fixed byte alphabet.

This policy constrains and renormalizes the decoder distribution at inference;
teacher-forced training uses all 258 logits. It adds UTF-8 and budget guarantees
to the v2 nested generation loop, without the author's later cache machinery.
"""

import codecs
import math

import torch

from ._records import (
    END_OF_DOCUMENT,
    END_OF_TOKEN,
    OUTPUT_SYMBOLS,
    ByteGenerationOptions,
    ByteTokenGenerationOutput,
)


def _validate_options(options):
    if not isinstance(options, ByteGenerationOptions):
        raise TypeError("options must be ByteGenerationOptions")
    if type(options.do_sample) is not bool:
        raise TypeError("do_sample must be a bool")
    if not math.isfinite(options.temperature) or options.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if options.top_k is not None and (
        type(options.top_k) is not int or not 1 <= options.top_k <= OUTPUT_SYMBOLS
    ):
        raise ValueError("top_k must be None or an integer from 1 through 258")
    if not math.isfinite(options.top_p) or not 0 < options.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if options.seed is not None and type(options.seed) is not int:
        raise TypeError("seed must be an integer or None")


def _allowed_symbols(data, remaining):
    decoder = codecs.getincrementaldecoder("utf-8")()
    decoder.decode(data, final=False)
    pending = decoder.getstate()[0]
    allowed = torch.zeros(OUTPUT_SYMBOLS, dtype=torch.bool)
    for value in range(256):
        probe = codecs.getincrementaldecoder("utf-8")()
        try:
            probe.decode(pending + bytes([value]), final=False)
        except UnicodeDecodeError:
            continue
        tail = probe.getstate()[0]
        required = 0
        if tail:
            width = 2 if tail[0] < 224 else 3 if tail[0] < 240 else 4
            required = width - len(tail)
        allowed[value] = required < remaining
    allowed[END_OF_TOKEN] = bool(data) and not pending
    allowed[END_OF_DOCUMENT] = not data
    return allowed


def _select_symbol(logits, allowed, options, generator):
    # A CPU generator has reproducible seeded sampling on every execution device.
    scores = logits.detach().float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("generation requires finite decoder logits")
    scores = (scores / options.temperature).masked_fill(~allowed, -torch.inf)
    if not options.do_sample:
        return int(scores.argmax())
    if options.top_k is not None:
        threshold = scores.topk(options.top_k).values[-1]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if options.top_p < 1:
        ordered, order = scores.sort(descending=True)
        cumulative = ordered.softmax(-1).cumsum(-1)
        remove = cumulative - ordered.softmax(-1) >= options.top_p
        scores[order[remove]] = -torch.inf
    return int(torch.multinomial(scores.softmax(-1), 1, generator=generator))


def _generate_token(model, conditioning, prefix, max_new_bytes, options, generator):
    _validate_options(options)
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a UTF-8 string")
    try:
        data = bytearray(prefix.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("prefix must be valid UTF-8") from error
    maximum = model.cfg.max_token_bytes
    if len(data) > maximum:
        raise ValueError("prefix exceeds max_token_bytes")
    if max_new_bytes is None:
        max_new_bytes = maximum
    if type(max_new_bytes) is not int or max_new_bytes < 0:
        raise ValueError("max_new_bytes must be a non-negative integer")
    if not isinstance(conditioning, torch.Tensor) or conditioning.shape != (
        model.cfg.conditioning_dim,
    ):
        raise ValueError("conditioning must have shape [conditioning_dim]")
    if generator is None and options.seed is not None:
        generator = torch.Generator().manual_seed(options.seed)
    initial_length = len(data)
    limit = min(maximum, initial_length + max_new_bytes)
    # A byte limit forces chunk completion without requiring a predicted EOW.
    stop_reason = "byte_limit" if limit == maximum else "byte_budget"
    training_states = [(child, child.training) for child in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            while len(data) < limit:
                prefixes = torch.tensor(
                    list(data), dtype=torch.long, device=conditioning.device
                ).reshape(1, 1, -1)
                lengths = prefixes.new_tensor([[len(data)]])
                output = model(conditioning.reshape(1, 1, -1), prefixes, lengths)
                symbol = _select_symbol(
                    output.logits[-1],
                    _allowed_symbols(data, limit - len(data)),
                    options,
                    generator,
                )
                if symbol >= END_OF_TOKEN:
                    stop_reason = (
                        "end_of_token" if symbol == END_OF_TOKEN else "end_of_document"
                    )
                    break
                data.append(symbol)
    finally:
        for child, was_training in training_states:
            child.training = was_training
    return ByteTokenGenerationOutput(
        data.decode("utf-8"), stop_reason, len(data) - initial_length
    )
