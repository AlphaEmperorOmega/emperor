import torch
from torch import Tensor


class Perplexity:
    """Derive bounded perplexity from one scalar token objective."""

    __MAXIMUM_TOKEN_LOSS = 20.0

    def from_token_loss(self, token_loss: Tensor) -> Tensor:
        if (
            not isinstance(token_loss, Tensor)
            or token_loss.numel() != 1
            or not token_loss.is_floating_point()
        ):
            raise ValueError(
                "Perplexity token loss must be a scalar floating-point tensor."
            )
        detached_loss = token_loss.detach().reshape(())
        if detached_loss.dtype in (torch.float16, torch.bfloat16):
            detached_loss = detached_loss.float()
        bounded_loss = detached_loss.clamp(max=self.__MAXIMUM_TOKEN_LOSS)
        return bounded_loss.exp()
