import asyncio

import pytest

from consultations import (
    ConsultationRequest,
    InMemoryConsultationRequestStore,
    InvalidConsultationRequestError,
    SqliteConsultationRequestStore,
)


def _request(*, session_id: str = "room-123") -> ConsultationRequest:
    return ConsultationRequest.create(
        session_id=session_id,
        name="  Rahul Sharma  ",
        phone="+91 (98765) 43210",
        request="  Website consultation  ",
    )


def test_request_is_normalized_and_has_stable_idempotency_key() -> None:
    first = _request()
    replay = ConsultationRequest.create(
        session_id="room-123",
        name="Rahul Sharma",
        phone="+91-98765-43210",
        request="Website consultation",
    )

    assert first.name == "Rahul Sharma"
    assert first.phone == "+919876543210"
    assert first.request == "Website consultation"
    assert first.request_id == replay.request_id


@pytest.mark.parametrize(
    ("name", "phone", "request_text"),
    [
        ("", "+919876543210", "Website consultation"),
        ("Rahul", "123", "Website consultation"),
        ("Rahul", "+919876543210", ""),
    ],
)
def test_request_rejects_incomplete_or_invalid_data(
    name: str,
    phone: str,
    request_text: str,
) -> None:
    with pytest.raises(InvalidConsultationRequestError):
        ConsultationRequest.create(
            session_id="room-123",
            name=name,
            phone=phone,
            request=request_text,
        )


@pytest.mark.asyncio
async def test_in_memory_store_is_idempotent_under_concurrent_retries() -> None:
    store = InMemoryConsultationRequestStore()
    request = _request()

    results = await asyncio.gather(*(store.create(request) for _ in range(20)))

    assert results.count(True) == 1
    assert results.count(False) == 19
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_sqlite_store_persists_and_deduplicates_retries(tmp_path) -> None:
    database_path = tmp_path / "brightpath.db"
    request = _request()

    first_process = SqliteConsultationRequestStore(database_path)
    assert await first_process.create(request) is True

    restarted_process = SqliteConsultationRequestStore(database_path)
    assert await restarted_process.create(request) is False
    assert await restarted_process.count() == 1

    stored = await restarted_process.get(request.request_id)
    assert stored == request
