from dataclasses import dataclass

SOURCE_COMMIT = "a3d2e0d26b56f3846f66a952536ffed4e401d05a"
SOURCE_BASE_URL = (
    f"https://raw.githubusercontent.com/multi30k/dataset/{SOURCE_COMMIT}/data/task1/raw"
)

VOCAB_SIZE = 8192
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MAX_SUBWORD_LENGTH = 6


@dataclass(frozen=True)
class Multi30kFile:
    split: str
    language: str
    filename: str
    sha256: str
    line_count: int

    @property
    def url(self) -> str:
        return f"{SOURCE_BASE_URL}/{self.filename}"

    @property
    def text_filename(self) -> str:
        return self.filename.removesuffix(".gz")


FILES: tuple[Multi30kFile, ...] = (
    Multi30kFile(
        "train",
        "de",
        "train.de.gz",
        "726e39b2fa9eb9ffb6dc763fb35a179f80fae06ffc5d28b6ace5faa883de28a6",
        29_000,
    ),
    Multi30kFile(
        "train",
        "en",
        "train.en.gz",
        "d79cfa999dd4c51d2cb42499b6796d5a882c3a8a961923c25a898c90f8bbd56f",
        29_000,
    ),
    Multi30kFile(
        "val",
        "de",
        "val.de.gz",
        "f0cba2f995189cf5770f29a8a9a537a3ad3f51657ad873405082ff6863a5e75a",
        1_014,
    ),
    Multi30kFile(
        "val",
        "en",
        "val.en.gz",
        "14f7d25ddd868909a9213e361768460edcacdd6ab9d1e77b92560dc10c10dc28",
        1_014,
    ),
    Multi30kFile(
        "test",
        "de",
        "test_2016_flickr.de.gz",
        "9204244e408ccb38d2a55cfcd344df15005fc42a07a6e55ca6c52b6ababb8cc8",
        1_000,
    ),
    Multi30kFile(
        "test",
        "en",
        "test_2016_flickr.en.gz",
        "611d361c6334bc7246101d48097c13cf5c4413c5befc793cc629934359d532d9",
        1_000,
    ),
)
