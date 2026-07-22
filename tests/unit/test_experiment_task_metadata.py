import unittest

from emperor.experiments import ExperimentTask
from model_runtime.packages import ModelIdentity, ModelPackage
from models.catalog import model_package
from models.linears.linear import dataset_options
from models.linears.linear.presets import Experiment as LinearExperiment


class _InvalidMetadata:
    @property
    def dataset_options_by_task(self):
        raise ValueError("broken dataset metadata")


class _MetadataImportFailureAdapter:
    def load_metadata(self):
        raise ImportError("broken model metadata import")


class _MetadataValidationFailureAdapter:
    def load_metadata(self):
        return _InvalidMetadata()


class TestExperimentTaskMetadata(unittest.TestCase):
    def test_catalog_experiment_rejects_unsupported_task(self):
        with self.assertRaises(ValueError) as context:
            LinearExperiment(
                experiment_task=ExperimentTask.TEXT_TRANSLATION,
                model_package=model_package("linears/linear"),
            )

        message = str(context.exception)
        self.assertIn("Unknown experiment task", message)
        self.assertIn("linears/linear", message)
        self.assertIn("Valid tasks: image-classification", message)

    def test_supported_task_uses_only_its_dataset_metadata(self):
        experiment = LinearExperiment(
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            model_package=model_package("linears/linear"),
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
                _MetadataImportFailureAdapter,
                ImportError,
                "broken model metadata import",
            ),
            (
                _MetadataValidationFailureAdapter,
                ValueError,
                "broken dataset metadata",
            ),
        )

        for adapter_type, error_type, message in cases:
            with self.subTest(adapter_type=adapter_type.__name__):
                package = ModelPackage(
                    ModelIdentity("broken", "missing"),
                    adapter_type(),
                )
                with self.assertRaisesRegex(error_type, message):
                    LinearExperiment(
                        experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
                        model_package=package,
                    )


if __name__ == "__main__":
    unittest.main()
