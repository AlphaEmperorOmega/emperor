import copy
import importlib
import io
import tempfile
import unittest
from dataclasses import replace

import pytest
import torch
from datasets import Dataset
from lightning import Trainer

from emperor.decoding.hierarchical import (
    HierarchicalLanguageModelBatch,
    HierarchicalTextCodec,
)
from models.catalog import model_package

GPT_PACKAGES = (
    "gpt/expert_linear",
)


def configuration(package_name, **overrides):
    package = model_package(package_name)
    values = {
        "batch_size": 2,
        "sequence_length": 4,
        "hidden_dim": 8,
        "input_dim": 256,
        "output_dim": 258,
        "attn_num_heads": 2,
        "num_experts": 2,
        "top_k": 1,
        "sampler_normalize_probabilities_flag": False,
        "capacity_factor": 0.0,
        "expert_stack_hidden_dim": 8,
        "ff_stack_hidden_dim": 8,
        "router_stack_hidden_dim": 8,
        "stack_num_layers": 1,
        "stack_dropout_probability": 0.0,
        "embedding_dropout_probability": 0.0,
        "hierarchical_language_model_flag": True,
        "hierarchical_byte_embedding_dim": 8,
        "hierarchical_byte_num_heads": 2,
        "hierarchical_byte_feed_forward_dim": 16,
        "hierarchical_max_token_bytes": 8,
        "lm_head_weight_tying_flag": False,
        **overrides,
    }
    if not package_name.split("/")[-1].startswith("expert_"):
        for key in (
            "num_experts",
            "top_k",
            "sampler_normalize_probabilities_flag",
            "capacity_factor",
            "expert_stack_hidden_dim",
            "router_stack_hidden_dim",
        ):
            values.pop(key, None)
    module = importlib.import_module(
        "models." + package_name.replace("/", ".") + ".config_builder"
    )
    builder_name = "".join(
        part.title() for part in package_name.split("/")[-1].split("_")
    )
    builder = getattr(module, f"Gpt{builder_name}ConfigBuilder")
    return package, builder(runtime=package.bind_runtime_defaults(values)).build()


class GptHierarchicalLanguageModelTests(unittest.TestCase):
    def test_hierarchical_mode_preserves_each_packages_backbone(self):
        from emperor.augmentations.adaptive_parameters import GeneratorDynamicBiasConfig
        from emperor.transformer import TransformerDecoderLayerState

        for name in GPT_PACKAGES:
            with self.subTest(package=name):
                torch.manual_seed(31)
                options = dict(stack_num_layers=2, num_experts=3, top_k=2)
                if name.endswith("_adaptive"):
                    options.update(
                        bias_option_flag=True,
                        bias_option=GeneratorDynamicBiasConfig,
                        adaptive_generator_stack_hidden_dim=8,
                    )
                package, cfg = configuration(name, **options)
                _, ordinary = configuration(
                    name, hierarchical_language_model_flag=False, **options
                )
                original = copy.deepcopy(cfg)
                composition = cfg.experiment_config.hierarchical_language_model_config
                self.assertEqual(
                    composition.backbone_config,
                    ordinary.experiment_config.decoder_config,
                )
                self.assertIsNot(
                    composition.backbone_config, cfg.experiment_config.decoder_config
                )
                model = package.build_model(cfg).eval()
                backbone = model.hierarchical_model.backbone
                reference = ordinary.experiment_config.decoder_config.build().eval()
                reference.load_state_dict(backbone.state_dict(), strict=True)
                hidden = torch.randn(2, 3, cfg.hidden_dim)
                actual = backbone(TransformerDecoderLayerState(hidden=hidden.clone()))
                expected = reference(
                    TransformerDecoderLayerState(hidden=hidden.clone())
                )
                torch.testing.assert_close(actual.hidden, expected.hidden)
                torch.testing.assert_close(actual.loss, expected.loss)
                if "/expert_" in name:
                    self.assertGreater(actual.loss.item(), 0)
                self.assertEqual(cfg, original)

    def test_public_shape_inspection_uses_raw_batch(self):
        from model_runtime.inspection import InspectionRequest, inspect_model_shapes

        for name in GPT_PACKAGES:
            result, trace = inspect_model_shapes(
                model_package(name),
                InspectionRequest(
                    preset="hierarchical",
                    dataset="wiki-text103-hierarchical",
                    overrides={
                        "hidden_dim": 8,
                        "batch_size": 1,
                        "sequence_length": 2,
                        "stack_num_layers": 1,
                        "attn_num_heads": 2,
                        **(
                            {"num_experts": 3, "top_k": 2} if "/expert_" in name else {}
                        ),
                        "hierarchical_byte_embedding_dim": 8,
                        "hierarchical_byte_num_heads": 2,
                        "hierarchical_max_token_bytes": 8,
                    },
                ),
            )
            self.assertEqual(result.identity.catalog_key, name)
            root = next(
                module for module in trace.modules if module.node_id == "__root__"
            )
            self.assertIn(
                (4, 258), [tuple(shape.shape) for shape in root.calls[0].outputs]
            )

    def test_training_rejects_misaligned_labels_and_handles_all_masked(self):
        package, cfg = configuration("gpt/expert_linear")
        model = package.build_model(cfg)
        batch = HierarchicalLanguageModelBatch.collate(
            list(HierarchicalTextCodec(8).training_windows("mat", 4))
        )
        labels = batch.labels.clone()
        labels[0] = 100
        with self.assertRaisesRegex(ValueError, "align"):
            model._model_step(replace(batch, labels=labels))
        masked = replace(
            batch,
            attention_mask=torch.zeros_like(batch.attention_mask),
            bos_mask=torch.zeros_like(batch.bos_mask),
            labels=torch.empty(0, dtype=torch.long),
            byte_count=0,
        )
        loss = model._model_step(masked)
        self.assertEqual(loss.item(), 0)
        loss.backward()

    def test_checkpoint_shapes_and_parameters_do_not_depend_on_vocabulary(self):
        from emperor.decoding.hierarchical import ByteGenerationOptions

        for name in GPT_PACKAGES:
            package, cfg = configuration(name)
            model = package.build_model(cfg).eval()
            original = copy.deepcopy(cfg)
            state = model.state_dict()
            recovered = package.checkpoint_config_overrides(
                {key: tuple(value.shape) for key, value in state.items()}
            )
            self.assertTrue(recovered["hierarchical_language_model_flag"])
            self.assertEqual(recovered["hierarchical_byte_embedding_dim"], 8)
            self.assertEqual(recovered["hierarchical_max_token_bytes"], 8)
            changed = copy.deepcopy(cfg)
            changed.input_dim = changed.output_dim = 267735
            other = package.build_model(changed).eval()
            self.assertEqual(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in other.parameters()),
            )
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            other.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            batch = HierarchicalLanguageModelBatch.collate(
                list(HierarchicalTextCodec(8).training_windows("Novel 🌍", 4))
            )
            torch.testing.assert_close(model(batch).logits, other(batch).logits)
            self.assertEqual(original, cfg)
            self.assertEqual(
                {key: tuple(value.shape) for key, value in model.state_dict().items()},
                {key: tuple(value.shape) for key, value in state.items()},
            )
            options = ByteGenerationOptions(do_sample=True, seed=44, top_k=4)
            first = model.generate_text(
                "unfin", max_new_bytes=6, max_new_tokens=3, options=options
            )
            second = model.generate_text(
                "unfin", max_new_bytes=6, max_new_tokens=3, options=options
            )
            self.assertEqual(first, second)
            self.assertTrue(first.text.startswith("unfin"))

    def test_cli_dataset_delivery_and_synthetic_inspection(self):
        from emperor.datasets.text.language_modeling import (
            WikiText2,
            WikiText103Hierarchical,
        )
        from emperor.experiments import ExperimentTask
        from model_runtime.inspection import configuration_schema
        from model_runtime.task_behavior import experiment_task_behavior
        from models.cli_selection import resolve_cli_selection
        from models.experiment_cli_parser import get_experiment_parser

        for name in GPT_PACKAGES:
            package, cfg = configuration(name)
            parser = get_experiment_parser(package)
            args = parser.parse_args(
                [
                    "--preset",
                    "hierarchical",
                    "--config",
                    "--hierarchical-byte-embedding-dim",
                    "16",
                ]
            )
            selected = resolve_cli_selection(args, package, package.preset_type)
            self.assertEqual(
                selected.config_overrides["hierarchical_byte_embedding_dim"], 16
            )
            self.assertIn(
                "HIERARCHICAL_LANGUAGE_MODEL_FLAG",
                {field.key for field in configuration_schema(package).fields},
            )
            behavior = experiment_task_behavior(ExperimentTask.CAUSAL_LANGUAGE_MODELING)
            self.assertEqual(
                behavior.dataset_constructor_kwargs(cfg)["max_token_bytes"], 8
            )
            sample = behavior.synthetic_inputs(WikiText103Hierarchical, cfg)
            self.assertIsInstance(sample[0], HierarchicalLanguageModelBatch)
            output = package.build_model(cfg)(*sample)
            self.assertEqual(output.logits.shape[-1], 258)
            with self.assertRaisesRegex(ValueError, "dataset|WikiText103Hierarchical"):
                package.build_configuration(package.preset_type.HIERARCHICAL, WikiText2)
            with self.assertRaisesRegex(ValueError, "raw-text dataset"):
                package.build_configuration(
                    package.preset_type.BASELINE,
                    WikiText103Hierarchical,
                    config_overrides={"lm_head_weight_tying_flag": False},
                )


    @pytest.mark.training
    def test_bounded_fixture_training_validation_and_generation(self):
        from emperor.datasets.text.language_modeling import WikiText103Hierarchical

        source = Dataset.from_dict({"text": [" = Tiny = \n", "mat café 😀\n"]})

        class LocalWikiText(WikiText103Hierarchical):
            def _dataset(self, split):
                return source

        for name in GPT_PACKAGES:
            package, cfg = configuration(name)
            model = package.build_model(cfg)
            with tempfile.TemporaryDirectory() as root:
                data = LocalWikiText(
                    root=root,
                    batch_size=2,
                    sequence_length=4,
                    max_token_bytes=8,
                    num_workers=0,
                )
                trainer = Trainer(
                    default_root_dir=root,
                    accelerator="cpu",
                    devices=1,
                    max_steps=1,
                    limit_val_batches=1,
                    num_sanity_val_steps=0,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )
                trainer.fit(model, datamodule=data)
                result = trainer.validate(model, datamodule=data, verbose=False)[0]
                self.assertTrue(
                    torch.isfinite(torch.tensor(result["validation/bits_per_byte"]))
                )
                self.assertNotIn("validation/perplexity", result)
                model.train()
                generated = model.generate_text(
                    "café ", max_new_tokens=2, max_new_bytes=8
                )
                self.assertTrue(generated.text.startswith("café "))
                generated.text.encode("utf-8", "strict")
                self.assertTrue(model.training)

    def test_all_gpt_packages_predict_bytes_without_vocabulary_binding(self):
        for package_name in GPT_PACKAGES:
            with self.subTest(package=package_name):
                package, cfg = configuration(package_name)
                model = package.build_model(cfg).eval()
                batch = HierarchicalLanguageModelBatch.collate(
                    list(HierarchicalTextCodec(8).training_windows("mat cat ", 4))
                )
                output = model(batch)
                self.assertEqual(output.logits.shape, (11, 258))
                self.assertFalse(hasattr(model, "lm_head"))
                self.assertFalse(hasattr(model, "token_text_adapter"))
                self.assertFalse(hasattr(model, "token_embedding"))
                loss = model._model_step(batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(
                    model.hierarchical_model.beginning_of_document.grad.abs().sum(), 0
                )
