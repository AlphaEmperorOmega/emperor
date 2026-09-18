"""Vocabulary-bound token spellings for the contextual embedding boundary."""

from collections.abc import Sequence

from torch import Tensor, nn


class TokenTextAdapter(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.token_texts: tuple[str, ...] | None = None

    def bind(self, token_texts: Sequence[str]) -> None:
        if isinstance(token_texts, (str, bytes)) or not isinstance(
            token_texts, Sequence
        ):
            raise TypeError(
                "Token vocabulary must be a sequence of token strings in ID order."
            )
        if len(token_texts) != self.vocabulary_size:
            raise ValueError(
                f"Token vocabulary must contain exactly {self.vocabulary_size} entries."
            )
        if any(type(token) is not str for token in token_texts):
            raise TypeError("Every token vocabulary entry must be an exact str.")
        tokens = tuple(token_texts)
        if self.token_texts is not None and self.token_texts != tokens:
            raise ValueError(
                "Token vocabulary differs from the bound/checkpoint vocabulary."
            )
        self.token_texts = tokens

    def bind_datamodule(self, datamodule) -> None:
        vocabulary = getattr(datamodule, "vocab", None)
        if vocabulary is None:
            raise ValueError(
                "Contextual embedding requires a prepared datamodule vocabulary."
            )
        if hasattr(vocabulary, "get_itos"):
            tokens = vocabulary.get_itos()
        elif hasattr(vocabulary, "itos"):
            tokens = vocabulary.itos
        elif hasattr(vocabulary, "lookup_token"):
            tokens = [
                vocabulary.lookup_token(index) for index in range(len(vocabulary))
            ]
        else:
            raise TypeError(
                "Datamodule vocabulary must expose token strings in ID order."
            )
        self.bind(tokens)

    def forward(self, input_ids: Tensor) -> list[list[str]]:
        if self.token_texts is None:
            raise ValueError(
                "Contextual embedding token vocabulary is unbound. Call "
                "model.set_token_vocabulary(tokens_in_id_order) for standalone use; "
                "Trainer setup binds the datamodule vocabulary automatically."
            )
        return [
            [self.token_texts[index] for index in row]
            for row in input_ids.detach().cpu().tolist()
        ]

    def get_extra_state(self):
        return {"schema_version": 1, "token_texts": self.token_texts}

    def set_extra_state(self, state) -> None:
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "token_texts",
        }:
            raise ValueError("Invalid contextual token vocabulary checkpoint state.")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError(
                "Unsupported contextual token vocabulary checkpoint schema."
            )
        if state["token_texts"] is not None:
            self.bind(state["token_texts"])
        elif self.token_texts is not None:
            raise ValueError(
                "Cannot restore an unbound vocabulary over a bound vocabulary."
            )
