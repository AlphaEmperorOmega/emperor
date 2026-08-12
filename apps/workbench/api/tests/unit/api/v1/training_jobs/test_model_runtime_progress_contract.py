from __future__ import annotations

import unittest

from model_runtime.runs.progress import (
    MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS,
    MODEL_RUNTIME_PROGRESS_EVENT_FIELDS,
    MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS,
)

from emperor_workbench.api.v1.training_jobs._contracts import (
    TrainingClusterInitializedProgressEventResponse,
    TrainingDatasetCompletedProgressEventResponse,
    TrainingDatasetStartedProgressEventResponse,
    TrainingErrorProgressEventResponse,
    TrainingNeuronAddedProgressEventResponse,
    TrainingNeuronsAddedProgressEventResponse,
    TrainingProgressEventBaseResponse,
    TrainingRunProgressEventResponse,
)


class ModelRuntimeProgressContractTests(unittest.TestCase):
    def test_api_contract_declares_every_runtime_owned_wire_field(self) -> None:
        contract_by_event = {
            "dataset_started": TrainingDatasetStartedProgressEventResponse,
            "dataset_completed": TrainingDatasetCompletedProgressEventResponse,
            "error": TrainingErrorProgressEventResponse,
            "epoch_started": TrainingRunProgressEventResponse,
            "step": TrainingRunProgressEventResponse,
            "validation": TrainingRunProgressEventResponse,
            "fit_completed": TrainingRunProgressEventResponse,
            "test_completed": TrainingRunProgressEventResponse,
            "cluster_initialized": TrainingClusterInitializedProgressEventResponse,
            "neuron_added": TrainingNeuronAddedProgressEventResponse,
            "neurons_added": TrainingNeuronsAddedProgressEventResponse,
        }

        self.assertEqual(
            set(contract_by_event), set(MODEL_RUNTIME_PROGRESS_EVENT_FIELDS)
        )
        self.assertTrue(
            set(MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS)
            <= set(TrainingProgressEventBaseResponse.model_fields)
        )
        for event_type, wire_fields in MODEL_RUNTIME_PROGRESS_EVENT_FIELDS.items():
            with self.subTest(event_type=event_type):
                contract = contract_by_event[event_type]
                self.assertTrue(set(wire_fields) <= set(contract.model_fields))
                for optional_field in MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS[
                    event_type
                ]:
                    self.assertFalse(
                        contract.model_fields[optional_field].is_required()
                    )


if __name__ == "__main__":
    unittest.main()
