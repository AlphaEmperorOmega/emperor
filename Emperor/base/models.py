# Extrnal
import torch.nn.functional as F
from ..library.choice import Library as L

# Local
from .utils import Module
from .utils import reshape, astype, argmax, float32, reduce_mean, randn


class Classifier(Module):
    """The base class of classification models."""

    def validation_step(self, batch):
        Y_hat, auxilaryLoss = self(*batch[:-1])
        loss = self.loss(Y_hat, batch[-1])

        if auxilaryLoss is not None:
            loss += auxilaryLoss
        self.plot("loss", loss, train=False)
        self.plot("acc", self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1])).to("cpu")
        preds = astype(argmax(Y_hat, axis=1), Y.dtype)
        compare = astype(preds == reshape(Y, -1), float32)
        return reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,)).to(L.Device)
        loss = F.cross_entropy(Y_hat, Y, reduction="mean" if averaged else "none")
        return loss

    def layer_summary(self, X_shape):
        X = randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)
