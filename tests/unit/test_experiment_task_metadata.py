import unittest

from emperor.experiments.tasks import ExperimentTask
from models.linears.linear import dataset_options
from models.linears.linear.presets import Experiment as LinearExperiment


class _InvalidMetadata:
    def dataset_options_for_task(self, task):
        raise ValueError("broken dataset metadata")


class _MetadataImportFailureExperiment(LinearExperiment):
    def _model_metadata(self):
        raise ImportError("broken model metadata import")


class _MetadataValidationFailureExperiment(LinearExperiment):
    def _model_metadata(self):
        return _InvalidMetadata()


class TestExperimentTaskMetadata(unittest.TestCase):
    def test_catalog_experiment_rejects_unsupported_task(self):
        with self.assertRaises(ValueError) as context:
            LinearExperiment(experiment_task=ExperimentTask.TEXT_TRANSLATION)

        message = str(context.exception)
        self.assertIn("Unknown experiment task", message)
        self.assertIn("linears/linear", message)
        self.assertIn("Valid tasks: image-classification", message)

    def test_supported_task_uses_only_its_dataset_metadata(self):
        experiment = LinearExperiment(
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION
        )

        self.assertEqual(
            experiment.experiment_task,
            ExperimentTask.IMAGE_CLASSIFICATION,
        )
        self.assertEqual(
            experiment.dataset_options,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                ExperimentTask.IMAGE_CLASSIFICATION
            ],
        )

    def test_metadata_failures_are_not_converted_to_image_defaults(self):
        cases = (
            (
                _MetadataImportFailureExperiment,
                ImportError,
                "broken model metadata import",
            ),
            (
                _MetadataValidationFailureExperiment,
                ValueError,
                "broken dataset metadata",
            ),
        )

        for experiment_type, error_type, message in cases:
            with self.subTest(experiment_type=experiment_type.__name__):
                with self.assertRaisesRegex(error_type, message):
                    experiment_type(
                        experiment_task=ExperimentTask.IMAGE_CLASSIFICATION
                    )


if __name__ == "__main__":
    unittest.main()
