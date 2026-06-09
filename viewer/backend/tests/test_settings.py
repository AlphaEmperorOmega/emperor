from __future__ import annotations

import os
import unittest
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import ValidationError

from viewer.backend.core.config import (
    LOCAL_FRONTEND_ORIGINS,
    ViewerApiSettings,
    get_viewer_api_settings,
)

SETTINGS_ENV_NAMES = (
    "VIEWER_API_AUTH_MODE",
    "VIEWER_API_TOKEN",
    "VIEWER_API_CORS_ORIGINS",
)


@contextmanager
def isolated_settings_env(**values: str) -> Iterator[None]:
    original = {name: os.environ.get(name) for name in SETTINGS_ENV_NAMES}
    try:
        for name in SETTINGS_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(values)
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ViewerApiSettingsTests(unittest.TestCase):
    def test_legacy_settings_module_reexports_canonical_settings(self) -> None:
        from viewer.backend import settings as legacy_settings

        self.assertIs(legacy_settings.LOCAL_FRONTEND_ORIGINS, LOCAL_FRONTEND_ORIGINS)
        self.assertIs(legacy_settings.ViewerApiSettings, ViewerApiSettings)
        self.assertIs(
            legacy_settings.get_viewer_api_settings,
            get_viewer_api_settings,
        )

    def test_defaults_keep_local_development_unauthenticated(self) -> None:
        with isolated_settings_env():
            settings = ViewerApiSettings()

        self.assertEqual(settings.auth_mode, "none")
        self.assertIsNone(settings.token)

    def test_defaults_keep_local_development_cors_origins(self) -> None:
        with isolated_settings_env():
            settings = ViewerApiSettings()

        self.assertEqual(settings.cors_origins, LOCAL_FRONTEND_ORIGINS)
        self.assertIsNot(settings.cors_origins, LOCAL_FRONTEND_ORIGINS)

    def test_bearer_mode_accepts_non_empty_token(self) -> None:
        settings = ViewerApiSettings(auth_mode="bearer", token="secret-token")

        self.assertEqual(settings.auth_mode, "bearer")
        self.assertEqual(settings.token, "secret-token")

    def test_bearer_mode_rejects_missing_or_empty_token(self) -> None:
        for token in (None, "", "   "):
            with self.subTest(token=token):
                kwargs = {"auth_mode": "bearer"}
                if token is not None:
                    kwargs["token"] = token

                with self.assertRaises(ValidationError):
                    ViewerApiSettings(**kwargs)

    def test_invalid_auth_mode_rejects(self) -> None:
        with self.assertRaises(ValidationError):
            ViewerApiSettings(auth_mode="basic", token="secret-token")

    def test_env_parses_auth_mode_and_token(self) -> None:
        with isolated_settings_env(
            VIEWER_API_AUTH_MODE="bearer",
            VIEWER_API_TOKEN="env-secret",
        ):
            settings = ViewerApiSettings()

        self.assertEqual(settings.auth_mode, "bearer")
        self.assertEqual(settings.token, "env-secret")

    def test_env_parses_cors_origins_json_array(self) -> None:
        origins = [
            "https://viewer.example.com",
            "https://admin.example.com",
        ]

        with isolated_settings_env(
            VIEWER_API_CORS_ORIGINS=(
                '["https://viewer.example.com","https://admin.example.com"]'
            ),
        ):
            settings = ViewerApiSettings()

        self.assertEqual(settings.cors_origins, origins)


if __name__ == "__main__":
    unittest.main()
