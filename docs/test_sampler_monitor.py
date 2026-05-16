import torch
import unittest

from torch import nn

from emperor.sampler.model import SamplerModel
from emperor.sampler.core.config import SamplerConfig
from emperor.sampler.core.monitor import SamplerMonitorCallback
from emperor.sampler.core.tracker import SamplerUsageTrackerManager


class FakeExperiment:
    def __init__(self):
        self.histograms = []
        self.images = []

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values.clone(), step))

    def add_image(self, tag, image, step, dataformats):
        self.images.append((tag, image.clone(), step, dataformats))


class FakeLogger:
    def __init__(self, experiment):
        self.experiment = experiment


class FakeLightningModule(nn.Module):
    def __init__(self, sampler: SamplerModel, experiment=None, global_step: int = 0):
        super().__init__()
        self.sampler = sampler
        self.logger = FakeLogger(experiment) if experiment is not None else None
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


class TestSamplerMonitorCallback(unittest.TestCase):
    def sampler_config(self, **overrides) -> SamplerConfig:
        values = {
            "top_k": 2,
            "threshold": 0.0,
            "filter_above_threshold": False,
            "num_topk_samples": 0,
            "normalize_probabilities_flag": False,
            "noisy_topk_flag": False,
            "num_experts": 4,
            "coefficient_of_variation_loss_weight": 0.0,
            "switch_loss_weight": 0.0,
            "zero_centred_loss_weight": 0.0,
            "mutual_information_loss_weight": 0.0,
            "router_config": None,
        }
        values.update(overrides)
        return SamplerConfig(**values)

    def build_module_with_sampler(
        self,
        experiment=None,
        global_step: int = 0,
        **sampler_overrides,
    ) -> tuple[FakeLightningModule, SamplerModel]:
        sampler = SamplerModel(self.sampler_config(**sampler_overrides))
        module = FakeLightningModule(
            sampler, experiment=experiment, global_step=global_step
        )
        return module, sampler

    def primed_callback(self, module, **callback_kwargs) -> SamplerMonitorCallback:
        callback = SamplerMonitorCallback(**callback_kwargs)
        callback.on_fit_start(trainer=None, pl_module=module)
        return callback

    def feed_sampler(self, sampler: SamplerModel, batch_size: int = 5) -> None:
        logits = torch.randn(batch_size, sampler.num_experts)
        sampler.sample_probabilities_and_indices(logits)

    def test_on_fit_start_attaches_trackers_to_sampler_modules(self):
        module, sampler = self.build_module_with_sampler()

        callback = self.primed_callback(module)

        self.assertIsNotNone(sampler.usage_tracker)
        self.assertEqual(len(callback._sampler_modules), 1)
        attached_name, attached_sampler = callback._sampler_modules[0]
        self.assertIs(attached_sampler, sampler)
        self.assertEqual(attached_name, "sampler")
        self.assertIn(attached_name, callback._usage_history)
        self.assertIn(attached_name, callback._mass_history)

    def test_on_train_batch_end_skips_when_not_at_logging_interval(self):
        module, sampler = self.build_module_with_sampler()
        callback = self.primed_callback(module, log_every_n_steps=10)
        self.feed_sampler(sampler)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=3
        )

        self.assertEqual(module.logged_scalars, [])

    def test_on_train_batch_end_logs_aggregate_scalars(self):
        module, sampler = self.build_module_with_sampler()
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_sampler(sampler)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        scalar_names = {name for name, _ in module.logged_scalars}
        for prefix in ("batch", "cumulative"):
            for suffix in (
                "active_experts",
                "usage_entropy",
                "usage_coefficient_of_variation",
                "max_usage_fraction",
                "min_usage_fraction",
                "max_probability_mass",
                "min_probability_mass",
            ):
                self.assertIn(f"sampler/{prefix}/{suffix}", scalar_names)

    def test_per_expert_scalars_logged_when_enabled(self):
        module, sampler = self.build_module_with_sampler()
        callback = self.primed_callback(
            module, log_every_n_steps=1, log_per_expert_scalars=True
        )
        self.feed_sampler(sampler)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        scalar_names = {name for name, _ in module.logged_scalars}
        for expert_idx in range(sampler.num_experts):
            self.assertIn(
                f"sampler/batch/expert_{expert_idx}/usage_fraction", scalar_names
            )
            self.assertIn(
                f"sampler/batch/expert_{expert_idx}/probability_mass", scalar_names
            )

    def test_per_expert_scalars_skipped_by_default(self):
        module, sampler = self.build_module_with_sampler()
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_sampler(sampler)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        for name, _ in module.logged_scalars:
            self.assertNotIn("/expert_", name)

    def test_visual_summaries_skipped_when_no_experiment(self):
        module, sampler = self.build_module_with_sampler(experiment=None)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_sampler(sampler)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        self.assertEqual(callback._usage_history["sampler"], [])
        self.assertEqual(callback._mass_history["sampler"], [])

    def test_visual_summaries_log_histogram_and_heatmap(self):
        experiment = FakeExperiment()
        module, sampler = self.build_module_with_sampler(
            experiment=experiment, global_step=7
        )
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_sampler(sampler)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        histogram_tags = {tag for tag, _, _ in experiment.histograms}
        self.assertIn("sampler/histogram/usage_fraction", histogram_tags)
        self.assertIn("sampler/histogram/probability_mass", histogram_tags)

        image_tags = {tag for tag, _, _, _ in experiment.images}
        self.assertIn("sampler/heatmap/usage_fraction", image_tags)
        self.assertIn("sampler/heatmap/probability_mass", image_tags)

        for _, _, step in experiment.histograms:
            self.assertEqual(step, 7)

    def test_visual_history_bounded_by_history_size(self):
        experiment = FakeExperiment()
        module, sampler = self.build_module_with_sampler(experiment=experiment)
        callback = self.primed_callback(
            module, log_every_n_steps=1, history_size=3
        )

        for batch_idx in range(5):
            self.feed_sampler(sampler)
            callback.on_train_batch_end(
                trainer=None,
                pl_module=module,
                outputs=None,
                batch=None,
                batch_idx=batch_idx,
            )

        self.assertEqual(len(callback._usage_history["sampler"]), 3)
        self.assertEqual(len(callback._mass_history["sampler"]), 3)

    def test_skips_sampler_when_tracker_missing(self):
        module, sampler = self.build_module_with_sampler()
        callback = self.primed_callback(module, log_every_n_steps=1)
        SamplerUsageTrackerManager().detach(sampler)
        self.feed_sampler(sampler)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        self.assertEqual(module.logged_scalars, [])

    def test_on_fit_end_detaches_trackers_and_clears_state(self):
        module, sampler = self.build_module_with_sampler()
        callback = self.primed_callback(module)

        callback.on_fit_end(trainer=None, pl_module=module)

        self.assertIsNone(sampler.usage_tracker)
        self.assertEqual(callback._sampler_modules, [])
        self.assertEqual(callback._usage_history, {})
        self.assertEqual(callback._mass_history, {})
        self.assertIsNone(callback._tracker_manager)


if __name__ == "__main__":
    unittest.main()
