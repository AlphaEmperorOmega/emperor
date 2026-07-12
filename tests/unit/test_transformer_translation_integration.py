import itertools
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from emperor.datasets.text.translation import Multi30kDeEn
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from models.transformer.expert_linear.config_builder import (
    TransformerExpertLinearConfigBuilder,
)
from models.transformer.expert_linear.model import Model as ExpertLinearModel
from models.transformer.expert_linear_adaptive.config_builder import (
    TransformerExpertLinearAdaptiveConfigBuilder,
)
from models.transformer.expert_linear_adaptive.model import (
    Model as ExpertLinearAdaptiveModel,
)
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear.model import Model as LinearModel
from models.transformer.linear_adaptive.config_builder import (
    TransformerLinearAdaptiveConfigBuilder,
)
from models.transformer.linear_adaptive.model import Model as LinearAdaptiveModel


class TestTransformerTranslationIntegration(unittest.TestCase):
    def package_cases(self):
        return (
            (TransformerLinearConfigBuilder, LinearModel),
            (TransformerLinearAdaptiveConfigBuilder, LinearAdaptiveModel),
            (TransformerExpertLinearConfigBuilder, ExpertLinearModel),
            (
                TransformerExpertLinearAdaptiveConfigBuilder,
                ExpertLinearAdaptiveModel,
            ),
        )

    def preset(self, builder_type, model_type, *, batch_size=4, length=6):
        options = dict(
            batch_size=batch_size,
            model_dim=32,
            source_sequence_length=length,
            target_sequence_length=length,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=4,
            feed_forward_hidden_dim=64,
            dropout_probability=0.0,
        )
        if "Expert" in builder_type.__name__:
            options.update(expert_num_experts=4, expert_top_k=2)
        return model_type(builder_type(**options).build())

    def corpus_nll(self, model, dataloader):
        weighted_nll = 0.0
        token_count = 0
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                output = model._model_step_outputs(batch)
                valid_tokens = batch[1][:, 1:].ne(model.pad_token_id).sum().item()
                weighted_nll += output.nll.item() * valid_tokens
                token_count += valid_tokens
        return weighted_nll / token_count

    @unittest.skipUnless(
        os.environ.get("EMPEROR_RUN_TRANSLATION_OVERFIT") == "1",
        "Set EMPEROR_RUN_TRANSLATION_OVERFIT=1 for the learning acceptance test.",
    )
    def test_all_packages_overfit_and_greedily_reproduce_copy_translation(self):
        batch = torch.tensor(
            [
                [2, 20, 21, 22, 3, 0],
                [2, 23, 24, 25, 3, 0],
                [2, 26, 27, 28, 3, 0],
                [2, 29, 30, 31, 3, 0],
            ]
        )
        labels = batch[:, 1:]
        valid_tokens = labels.ne(0)
        for builder_type, model_type in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                torch.manual_seed(0)
                model = self.preset(builder_type, model_type)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                for _ in range(200):
                    logits, auxiliary_loss = model(batch, batch[:, :-1])
                    loss = (
                        F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=0,
                        )
                        + auxiliary_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                accuracy = (
                    (logits.argmax(dim=-1)[valid_tokens] == labels[valid_tokens])
                    .float()
                    .mean()
                )
                generated = model.generate(batch, max_length=batch.size(1))
                self.assertGreaterEqual(accuracy.item(), 0.98)
                torch.testing.assert_close(generated, batch)

    @unittest.skipUnless(
        os.environ.get("EMPEROR_RUN_MULTI30K_SMOKE") == "1",
        "Set EMPEROR_RUN_MULTI30K_SMOKE=1 for the real-corpus smoke test.",
    )
    def test_all_packages_train_validate_test_and_generate_on_real_multi30k(self):
        root = Path(
            os.environ.get(
                "EMPEROR_MULTI30K_ROOT",
                "/tmp/emperor-multi30k-integration",
            )
        )
        data = Multi30kDeEn(
            root=root,
            batch_size=4,
            source_sequence_length=12,
            target_sequence_length=12,
            num_workers=0,
            seed=0,
        )
        data.setup("fit")
        train_batches = list(itertools.islice(data.train_dataloader(), 2))
        validation_batches = list(itertools.islice(data.val_dataloader(), 2))
        data.setup("test")
        test_batches = list(itertools.islice(data.test_dataloader(), 2))

        for builder_type, model_type in self.package_cases():
            with self.subTest(package=builder_type.__name__):
                torch.manual_seed(0)
                model = self.preset(
                    builder_type,
                    model_type,
                    batch_size=4,
                    length=12,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                tracked_parameter = next(model.parameters())
                before = tracked_parameter.detach().clone()
                for batch in train_batches:
                    loss = model._model_step(batch)
                    self.assertTrue(torch.isfinite(loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.assertFalse(torch.equal(before, tracked_parameter.detach()))

                with torch.no_grad():
                    for batch in (*validation_batches, *test_batches):
                        self.assertTrue(torch.isfinite(model._model_step(batch)))
                    generated = model.generate(
                        test_batches[-1][0],
                        max_length=12,
                    )
                self.assertEqual(
                    generated.shape,
                    (test_batches[-1][0].size(0), 12),
                )
                self.assertTrue(torch.all(generated[:, 0] == model.bos_token_id))

    @unittest.skipUnless(
        os.environ.get("EMPEROR_RUN_TRANSLATION_QUALITY") == "1",
        "Set EMPEROR_RUN_TRANSLATION_QUALITY=1 for the full quality acceptance.",
    )
    def test_default_linear_reaches_multi30k_quality_threshold(self):
        seed_everything(0, workers=True)
        root = Path(
            os.environ.get(
                "EMPEROR_MULTI30K_ROOT",
                "/tmp/emperor-multi30k-integration",
            )
        )
        data = Multi30kDeEn(root=root, num_workers=0, seed=0)
        data.setup("fit")
        model = LinearModel(TransformerLinearConfigBuilder().build())
        untrained_validation_nll = self.corpus_nll(
            model,
            data.val_dataloader(),
        )
        trainer = Trainer(
            max_epochs=30,
            deterministic=True,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            callbacks=[
                EarlyStopping(
                    monitor="validation/loss",
                    patience=5,
                    mode="min",
                    check_finite=True,
                )
            ],
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, datamodule=data)
        data.setup("validate")
        trained_validation_nll = self.corpus_nll(
            model,
            data.val_dataloader(),
        )
        results = trainer.test(model, datamodule=data)

        self.assertLess(trained_validation_nll, untrained_validation_nll)
        self.assertGreaterEqual(float(results[0]["test/bleu"]), 0.10)


if __name__ == "__main__":
    unittest.main()
