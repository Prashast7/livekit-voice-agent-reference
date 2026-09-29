from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class InvalidConsultationRequestError(ValueError):
    """Raised when a caller request is incomplete or unsafe to persist."""


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _normalize_phone(value: str) -> str:
    stripped = value.strip()
    has_country_prefix = stripped.startswith("+")
    digits = re.sub(r"\D", "", stripped)
    if not 7 <= len(digits) <= 15:
        raise InvalidConsultationRequestError(
            "The phone number must contain between 7 and 15 digits."
        )
    return f"+{digits}" if has_country_prefix else digits


@dataclass(frozen=True, slots=True)
class ConsultationRequest:
    """A normalized callback request with a stable retry identifier."""

    request_id: str
    session_id: str
    name: str
    phone: str
    request: str
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        name: str,
        phone: str,
        request: str,
        created_at: datetime | None = None,
    ) -> ConsultationRequest:
        normalized_session_id = _collapse_whitespace(session_id)
        normalized_name = _collapse_whitespace(name)
        normalized_request = _collapse_whitespace(request)

        if not normalized_session_id:
            raise InvalidConsultationRequestError("A session identifier is required.")
        if not 2 <= len(normalized_name) <= 100:
            raise InvalidConsultationRequestError(
                "The caller name must contain between 2 and 100 characters."
            )
        if not 5 <= len(normalized_request) <= 1000:
            raise InvalidConsultationRequestError(
                "The request must contain between 5 and 1000 characters."
            )

        normalized_phone = _normalize_phone(phone)
        idempotency_material = "\x00".join(
            (normalized_session_id, normalized_phone, normalized_request.casefold())
        )
        request_id = hashlib.sha256(idempotency_material.encode()).hexdigest()

        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        return cls(
            request_id=request_id,
            session_id=normalized_session_id,
            name=normalized_name,
            phone=normalized_phone,
            request=normalized_request,
            created_at=timestamp,
        )


class ConsultationRequestStore(Protocol):
    """Persistence boundary for idempotent consultation requests."""

    async def create(self, request: ConsultationRequest) -> bool:
        """Persist a request, returning False when it was already processed."""


class InMemoryConsultationRequestStore:
    """Deterministic store for tests and local single-process development."""

    def __init__(self) -> None:
        self._requests: dict[str, ConsultationRequest] = {}
        self._lock = asyncio.Lock()

    async def create(self, request: ConsultationRequest) -> bool:
        async with self._lock:
            if request.request_id in self._requests:
                return False
            self._requests[request.request_id] = request
            return True

    async def get(self, request_id: str) -> ConsultationRequest | None:
        async with self._lock:
            return self._requests.get(request_id)

    async def count(self) -> int:
        async with self._lock:
            return len(self._requests)


class SqliteConsultationRequestStore:
    """Durable local adapter with a database-enforced idempotency key.

    SQLite is intentionally a development adapter. A horizontally scaled
    deployment should use the same store contract with a shared database.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS consultation_requests (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    request TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    async def create(self, request: ConsultationRequest) -> bool:
        return await asyncio.to_thread(self._create_sync, request)

    def _create_sync(self, request: ConsultationRequest) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO consultation_requests (
                    request_id,
                    session_id,
                    name,
                    phone,
                    request,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.session_id,
                    request.name,
                    request.phone,
                    request.request,
                    request.created_at.isoformat(),
                ),
            )
            return cursor.rowcount == 1

    async def get(self, request_id: str) -> ConsultationRequest | None:
        return await asyncio.to_thread(self._get_sync, request_id)

    def _get_sync(self, request_id: str) -> ConsultationRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_id, session_id, name, phone, request, created_at
                FROM consultation_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return ConsultationRequest(
            request_id=row["request_id"],
            session_id=row["session_id"],
            name=row["name"],
            phone=row["phone"],
            request=row["request"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM consultation_requests"
            ).fetchone()
        return int(row["total"])
