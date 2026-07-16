from __future__ import annotations

import unittest

from ._contract_support import (
    HIGH_RISK_SCHEMA_PARITY_GROUPS,
    INTENTIONAL_FRONTEND_DEFAULT_FIELDS,
    INTENTIONAL_FRONTEND_REQUIRED_FIELD_LOOSENESS,
    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA,
    SCHEMA_PARITY_BY_BACKEND_SCHEMA,
    SCHEMA_PARITY_CASES,
    _openapi_property_schema,
    _openapi_required_fields,
    schemas,
)


class ApiSchemaContractTests(unittest.TestCase):
    def test_schema_parity_case_inventory_is_unique(self) -> None:
        self.assertEqual(
            len(SCHEMA_PARITY_BY_BACKEND_SCHEMA),
            len(SCHEMA_PARITY_CASES),
        )

    def test_schema_parity_case_field_inventory_is_stable(self) -> None:
        for parity_case in SCHEMA_PARITY_CASES:
            with self.subTest(model=parity_case.backend_schema.__name__):
                backend_fields = set(parity_case.backend_fields)

                self.assertEqual(
                    tuple(parity_case.backend_schema.model_fields),
                    parity_case.backend_fields,
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            set(parity_case.frontend_required_fields) - backend_fields
                        )
                    ),
                    (),
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            set(parity_case.intentional_frontend_required_looseness)
                            - backend_fields
                        )
                    ),
                    (),
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            set(parity_case.intentional_frontend_default_fields)
                            - backend_fields
                        )
                    ),
                    (),
                )

    def test_schema_parity_schemas_reject_extra_fields(self) -> None:
        for parity_case in SCHEMA_PARITY_CASES:
            with self.subTest(model=parity_case.backend_schema.__name__):
                self.assertEqual(
                    parity_case.backend_schema.model_config.get("extra"),
                    "forbid",
                )

    def test_opaque_json_fields_use_named_json_openapi_schemas(self) -> None:
        expected_refs = {
            ("GraphConfigFieldResponse", "value"): "JsonValue",
            ("GraphNodeResponse", "details"): "JsonObject",
            ("LogRunResponse", "metrics"): "JsonObject",
            ("LogRunArtifactsResponse", "params"): "JsonObject",
            ("LogRunArtifactsResponse", "metrics"): "JsonObject",
            ("TrainingRunResponse", "metrics"): "JsonObject",
            ("TrainingJobResponse", "metrics"): "JsonObject",
        }

        for (schema_name, field_name), ref_name in expected_refs.items():
            with self.subTest(schema=schema_name, field=field_name):
                self.assertEqual(
                    _openapi_property_schema(schema_name, field_name),
                    {"$ref": f"#/components/schemas/{ref_name}"},
                )

    def test_high_risk_nested_schema_parity_groups_are_covered(self) -> None:
        for group, group_schemas in HIGH_RISK_SCHEMA_PARITY_GROUPS.items():
            with self.subTest(group=group):
                missing = [
                    schema.__name__
                    for schema in group_schemas
                    if schema not in SCHEMA_PARITY_BY_BACKEND_SCHEMA
                ]

                self.assertEqual(missing, [])

    def test_openapi_required_fields_have_frontend_required_parity(self) -> None:
        for (
            (
                backend_schema,
                frontend_contract,
            ),
            frontend_required_fields,
        ) in OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA.items():
            with self.subTest(
                backend_schema=backend_schema.__name__,
                frontend_contract=frontend_contract,
            ):
                frontend_required = set(frontend_required_fields)
                openapi_required = set(
                    _openapi_required_fields(backend_schema.__name__)
                )
                allowed_loose_fields = set(
                    INTENTIONAL_FRONTEND_REQUIRED_FIELD_LOOSENESS.get(
                        (backend_schema, frontend_contract),
                        {},
                    )
                )

                self.assertEqual(
                    tuple(sorted(frontend_required - set(backend_schema.model_fields))),
                    (),
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            openapi_required - frontend_required - allowed_loose_fields
                        )
                    ),
                    (),
                )

    def test_frontend_required_field_looseness_annotations_are_current(self) -> None:
        for (
            (
                backend_schema,
                frontend_contract,
            ),
            looseness_annotations,
        ) in INTENTIONAL_FRONTEND_REQUIRED_FIELD_LOOSENESS.items():
            with self.subTest(
                backend_schema=backend_schema.__name__,
                frontend_contract=frontend_contract,
            ):
                self.assertIn(
                    (backend_schema, frontend_contract),
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA,
                )
                frontend_required = set(
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA[
                        (backend_schema, frontend_contract)
                    ]
                )
                openapi_required = set(
                    _openapi_required_fields(backend_schema.__name__)
                )
                annotated_fields = set(looseness_annotations)

                self.assertEqual(
                    tuple(sorted(annotated_fields - openapi_required)),
                    (),
                )
                self.assertEqual(
                    tuple(sorted(annotated_fields & frontend_required)),
                    (),
                )
                self.assertTrue(
                    all(reason for reason in looseness_annotations.values())
                )

    def test_frontend_default_field_annotations_are_current(self) -> None:
        for (
            backend_schema,
            frontend_contract,
        ), default_annotations in INTENTIONAL_FRONTEND_DEFAULT_FIELDS.items():
            with self.subTest(
                backend_schema=backend_schema.__name__,
                frontend_contract=frontend_contract,
            ):
                self.assertIn(
                    (backend_schema, frontend_contract),
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA,
                )
                frontend_required = set(
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA[
                        (backend_schema, frontend_contract)
                    ]
                )

                for field_name, reason in default_annotations.items():
                    with self.subTest(field=field_name):
                        self.assertTrue(reason)
                        self.assertIn(field_name, backend_schema.model_fields)
                        self.assertNotIn(field_name, frontend_required)
                        self.assertFalse(
                            backend_schema.model_fields[field_name].is_required()
                        )

    def test_capabilities_schema_defaults_additive_fields(
        self,
    ) -> None:
        capabilities = schemas.CapabilitiesResponse(
            authMode="none",
            trainingEnabled=False,
            logDeletionEnabled=False,
        )

        self.assertEqual(capabilities.trainingCancellationCapability, "unsupported")
        self.assertFalse(capabilities.trainingResourceLimitsEnforced)
        self.assertEqual(capabilities.uploadsEnabled, False)
        self.assertIsNone(capabilities.maxUploadSize)
