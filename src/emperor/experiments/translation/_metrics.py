from torch import Tensor
from torchmetrics.text import SacreBLEUScore

from emperor.experiments._perplexity import Perplexity

from ._records import TranslationStepOutput

_PERPLEXITY = Perplexity()


def _translation_step_metrics(
    stage: str,
    output: TranslationStepOutput,
    pad_token_id: int,
) -> dict[str, Tensor]:
    valid_tokens = output.labels != pad_token_id
    predictions = output.logits.argmax(dim=-1)
    if bool(valid_tokens.any().item()):
        token_accuracy = (
            (predictions[valid_tokens] == output.labels[valid_tokens]).float().mean()
        )
    else:
        token_accuracy = output.logits.new_zeros(())
    perplexity = _PERPLEXITY.from_token_loss(output.nll)
    return {
        f"{stage}/loss": output.total_loss,
        f"{stage}/nll": output.nll,
        f"{stage}/perplexity": perplexity,
        f"{stage}/token_accuracy": token_accuracy,
        f"{stage}/auxiliary_loss": output.auxiliary_loss,
    }


def _corpus_bleu(
    predictions: list[str],
    references: list[str],
    device,
) -> Tensor:
    metric = SacreBLEUScore(tokenize="13a").to(device)
    return metric(predictions, [[reference] for reference in references])
