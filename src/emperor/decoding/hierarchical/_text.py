import re
from collections.abc import Iterator


class HierarchicalTextCodec:
    """V2-style whitespace spans with additional UTF-8-safe byte bounds.

    Trailing whitespace stays with its preceding span; leading whitespace forms
    its own span. This is not the later hat-splitter's Unicode word-boundary rule.
    """

    def __init__(self, max_token_bytes: int):
        if type(max_token_bytes) is not int or max_token_bytes <= 0:
            raise ValueError("max_token_bytes must be a positive integer")
        self.max_token_bytes = max_token_bytes

    def split_text(self, text: str) -> Iterator[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        for span in re.finditer(r"\S+\s*|\s+", text):
            characters = []
            byte_count = 0
            for character in span.group():
                try:
                    width = len(character.encode("utf-8"))
                except UnicodeEncodeError as error:
                    raise ValueError("text must be valid UTF-8") from error
                if width > self.max_token_bytes:
                    raise ValueError("a Unicode character exceeds max_token_bytes")
                if byte_count + width > self.max_token_bytes:
                    yield "".join(characters)
                    characters = []
                    byte_count = 0
                characters.append(character)
                byte_count += width
            if characters:
                yield "".join(characters)

    def training_windows(self, text: str, sequence_length: int) -> Iterator[dict]:
        """Yield shifted windows of one document, retaining its BOS and EOS pair."""
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        contexts, targets, bos = [], [], []
        previous = ""
        first = True
        for target in self.split_text(text):
            contexts.append(previous)
            targets.append(target)
            bos.append(first)
            previous, first = target, False
            if len(contexts) == sequence_length:
                yield {
                    "context_texts": contexts,
                    "target_texts": targets,
                    "bos_mask": bos,
                }
                contexts, targets, bos = [], [], []
        contexts.append(previous)
        targets.append(None)  # EOS has one control prediction and no trailing EOW.
        bos.append(first)
        yield {"context_texts": contexts, "target_texts": targets, "bos_mask": bos}
