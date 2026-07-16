from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from emperor_workbench.filesystem import apply_owner_only_permissions


@dataclass(frozen=True, slots=True)
class StoredMutationResponse:
    status_code: int
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes


@dataclass(frozen=True, slots=True)
class MutationAdmission:
    action: str
    response: StoredMutationResponse | None = None
    request_hash: str | None = None


class MutationJournal:
    """SQLite-backed journal with one atomic admission decision per scoped key."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "mutation-journal.sqlite3"
        self._owner_token = uuid.uuid4().hex

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        apply_owner_only_permissions(self.root)
        connection = sqlite3.connect(self.path, timeout=1.0)
        apply_owner_only_permissions(self.path)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mutation_journal (
                caller TEXT NOT NULL,
                method TEXT NOT NULL,
                route TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                owner TEXT NOT NULL,
                status_code INTEGER,
                headers_json TEXT,
                response_body BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (caller, method, route, idempotency_key)
            )
            """
        )
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def admit(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
    ) -> MutationAdmission:
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT request_hash, state, owner, status_code,
                       headers_json, response_body
                  FROM mutation_journal
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ?
                """,
                (caller, method, route, key),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO mutation_journal (
                        caller, method, route, idempotency_key, request_hash,
                        state, owner, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?)
                    """,
                    (
                        caller,
                        method,
                        route,
                        key,
                        "",
                        self._owner_token,
                        now,
                        now,
                    ),
                )
                return MutationAdmission("execute")

            stored_hash, state, owner, status_code, headers_json, body = row
            if state == "completed":
                decoded_headers = tuple(
                    (name.encode("latin-1"), value.encode("latin-1"))
                    for name, value in json.loads(headers_json or "[]")
                )
                return MutationAdmission(
                    "replay",
                    StoredMutationResponse(
                        status_code=int(status_code),
                        headers=decoded_headers,
                        body=bytes(body or b""),
                    ),
                    str(stored_hash),
                )
            if owner == self._owner_token:
                return MutationAdmission("in_progress")

            connection.execute(
                """
                UPDATE mutation_journal
                   SET owner = ?, updated_at = ?
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ?
                """,
                (self._owner_token, now, caller, method, route, key),
            )
            return MutationAdmission("execute")

    def record_request_hash(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
        request_hash: str,
    ) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT request_hash FROM mutation_journal
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND state = 'in_progress'
                   AND owner = ?
                """,
                (caller, method, route, key, self._owner_token),
            ).fetchone()
            if row is None:
                return False
            stored_hash = str(row[0] or "")
            if stored_hash and stored_hash != request_hash:
                return False
            connection.execute(
                """
                UPDATE mutation_journal
                   SET request_hash = ?, updated_at = ?
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND owner = ?
                """,
                (
                    request_hash,
                    time.time(),
                    caller,
                    method,
                    route,
                    key,
                    self._owner_token,
                ),
            )
            return True

    def complete(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
        response: StoredMutationResponse,
    ) -> None:
        headers_json = json.dumps(
            [
                [name.decode("latin-1"), value.decode("latin-1")]
                for name, value in response.headers
            ],
            separators=(",", ":"),
        )
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE mutation_journal
                   SET state = 'completed', status_code = ?, headers_json = ?,
                       response_body = ?, updated_at = ?
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND owner = ?
                """,
                (
                    response.status_code,
                    headers_json,
                    response.body,
                    time.time(),
                    caller,
                    method,
                    route,
                    key,
                    self._owner_token,
                ),
            )

    def abandon(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                DELETE FROM mutation_journal
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND state = 'in_progress'
                   AND owner = ?
                """,
                (caller, method, route, key, self._owner_token),
            )


__all__ = ["MutationAdmission", "MutationJournal", "StoredMutationResponse"]
