"""Public Interface for supported causal language-modeling datasets."""

from emperor.datasets.text.language_modeling._penn_treebank import PennTreebank
from emperor.datasets.text.language_modeling._wiki_text_2 import WikiText2

__all__ = ("PennTreebank", "WikiText2")
