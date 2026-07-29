from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class _TokenSpan:
    start: int
    end: int


class _AnswerSpanAligner:
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]],
        context_length: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._context_length = context_length

    def first_in_window(
        self,
        context: str,
        answer_starts: Sequence[int],
        answer_texts: Sequence[str],
    ) -> _TokenSpan | None:
        if len(answer_starts) != len(answer_texts):
            raise ValueError(
                "answer_start and text must contain the same number of answers"
            )

        context_tokens = self._tokenizer(context)
        for answer_start, answer_text in zip(
            answer_starts,
            answer_texts,
            strict=True,
        ):
            span = self._align_answer(
                context,
                context_tokens,
                answer_start,
                answer_text,
            )
            if span is not None and span.end < self._context_length:
                return span

        return None

    def _align_answer(
        self,
        context: str,
        context_tokens: list[str],
        answer_start: int,
        answer_text: str,
    ) -> _TokenSpan | None:
        answer_end = answer_start + len(answer_text)
        if not self._matches_source_span(
            context,
            answer_start,
            answer_end,
            answer_text,
        ):
            return None

        prefix_tokens = self._tokenizer(context[:answer_start])
        answer_tokens = self._tokenizer(context[answer_start:answer_end])
        suffix_tokens = self._tokenizer(context[answer_end:])
        if not answer_tokens:
            return None
        if prefix_tokens + answer_tokens + suffix_tokens != context_tokens:
            return None

        token_start = len(prefix_tokens)
        return _TokenSpan(
            start=token_start,
            end=token_start + len(answer_tokens) - 1,
        )

    @staticmethod
    def _matches_source_span(
        context: str,
        answer_start: int,
        answer_end: int,
        answer_text: str,
    ) -> bool:
        return (
            0 <= answer_start < answer_end <= len(context)
            and context[answer_start:answer_end] == answer_text
        )
