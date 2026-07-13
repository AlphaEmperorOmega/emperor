from __future__ import annotations

import unittest

from workbench.backend.api.v1.logs_mapping import (
    log_archive_import_to_payload,
    log_run_page_to_payload,
)
from workbench.backend.run_history.records import (
    LogArchiveImportResult,
    LogRun,
    LogRunExperimentFacets,
    LogRunFacets,
    LogRunFacetValue,
    LogRunModelFacet,
    LogRunPage,
)
from workbench.backend.schemas import LogArchiveImportResponse, LogRunsResponse


class LogsHttpMappingTests(unittest.TestCase):
    def test_run_page_mapping_owns_camel_case_and_model_identity_expansion(
        self,
    ) -> None:
        run = LogRun(
            id="run-1",
            group="group",
            experiment="experiment",
            model="linears/linear",
            preset="BASELINE",
            experiment_task="image-classification",
            dataset="Mnist",
            run_name="run_20260713_010203",
            timestamp="2026-07-13 01:02:03",
            version="version_0",
            relative_path="experiment/linears/linear/BASELINE/Mnist/run/version_0",
            has_result=True,
            event_file_count=1,
            checkpoint_count=2,
            has_hparams=True,
            metrics={"test/accuracy": 0.9},
        )
        page = LogRunPage(
            runs=(run,),
            total=1,
            limit=10,
            offset=0,
            has_more=False,
            facets=LogRunFacets(
                experiments=(
                    LogRunExperimentFacets(
                        experiment="experiment",
                        run_count=1,
                        datasets=(LogRunFacetValue("Mnist", 1),),
                        models=(LogRunModelFacet("linears/linear", 1),),
                        presets=(LogRunFacetValue("BASELINE", 1),),
                    ),
                )
            ),
        )

        payload = log_run_page_to_payload(page)
        response = LogRunsResponse.model_validate(payload)

        self.assertEqual(payload["runs"][0]["modelType"], "linears")
        self.assertEqual(payload["runs"][0]["model"], "linear")
        self.assertEqual(payload["runs"][0]["runName"], run.run_name)
        self.assertEqual(payload["runs"][0]["relativePath"], run.relative_path)
        self.assertEqual(response.facets.experiments[0].models[0].modelType, "linears")

    def test_archive_import_mapping_preserves_contract_fields(self) -> None:
        value = LogArchiveImportResult(
            extracted_file_count=3,
            skipped_file_count=1,
            destination_root="/logs",
        )

        payload = log_archive_import_to_payload(value)
        response = LogArchiveImportResponse.model_validate(payload)

        self.assertEqual(response.extractedFileCount, 3)
        self.assertEqual(response.skippedFileCount, 1)
        self.assertEqual(response.destinationRoot, "/logs")


if __name__ == "__main__":
    unittest.main()
