from collections.abc import Sequence


class _NERSchema:
    _PAD_TOKEN = "<pad>"
    _UNKNOWN_TOKEN = "<unk>"
    _PAD_INDEX = 0
    _UNKNOWN_INDEX = 1

    def __init__(self, training_samples: Sequence[dict]) -> None:
        vocabulary = sorted(
            {
                token
                for sample in training_samples
                for token in sample["tokens"]
            }
        )
        self._tokens = (self._PAD_TOKEN, self._UNKNOWN_TOKEN, *vocabulary)
        self._token_to_index = {
            token: index for index, token in enumerate(self._tokens)
        }

    @property
    def fingerprint(self) -> tuple[str, ...]:
        return self._tokens

    def encode(self, tokens: Sequence[str], sequence_length: int) -> list[int]:
        encoded = [
            self._token_to_index.get(token, self._UNKNOWN_INDEX)
            for token in tokens[:sequence_length]
        ]
        return encoded + [self._PAD_INDEX] * (sequence_length - len(encoded))
