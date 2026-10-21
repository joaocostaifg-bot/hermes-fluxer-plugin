"""Current-review edge cases: sticky fallback, reader retry, handoff limits."""

from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


class Titles:
    def find_session_by_origin(self, **kwargs):
        return "synthetic-session"

    def get_session_title_source(self, session_id):
        return "user"

    def get_session_title(self, session_id):
        return "synthetic title"


def make_adapter():
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "thread_replies": True,
        "delivery_verification": False, "allow_all_users": True,
        "require_mention": False, "gateway_state_updates": False,
    }))
    adapter._request = AsyncMock(return_value={"id": "synthetic-output"})
    adapter._multipart_request = AsyncMock(return_value={"id": "synthetic-file"})
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", [RuntimeError("temporary rejection"), None, {}])
async def test_failed_creation_keeps_parent_for_text_and_upload_but_not_next_turn(
    monkeypatch, tmp_path, first_failure,
):
    monkeypatch.setattr(fluxer_adapter, "_open_session_store", lambda home=None: Titles())
    adapter = make_adapter()
    adapter.create_thread = AsyncMock(side_effect=[first_failure, {"id": "next-thread"}])
    attachment = tmp_path / "note.txt"
    attachment.write_text("synthetic content")
    assert (await adapter.send("channel", "progress", reply_to="first-trigger")).success
    assert (await adapter.send("channel", "final", reply_to="first-trigger")).success
    assert (await adapter.send_document("channel", str(attachment), reply_to="first-trigger")).success
    assert adapter.create_thread.await_count == 1
    assert [call.args[1] for call in adapter._request.await_args_list] == [
        "/channels/channel/messages", "/channels/channel/messages",
    ]
    assert adapter._multipart_request.await_args.args[1] == "/channels/channel/messages"
    assert (await adapter.send("channel", "next turn", reply_to="next-trigger")).success
    assert adapter._request.await_args.args[1] == "/channels/next-thread/messages"


class _RaceAlreadyCreated(RuntimeError):
    def __init__(self):
        import httpx

        request = httpx.Request(
            "POST", "https://fluxer.test/api/v1/channels/channel/messages/anchor/threads"
        )
        response = httpx.Response(
            400,
            request=request,
            json={
                "code": "THREAD_ALREADY_CREATED_FOR_MESSAGE",
                "message": "A thread has already been created for this message.",
            },
        )
        super().__init__("already created")
        self.response = response


@pytest.mark.asyncio
async def test_thread_create_race_adopts_existing_thread(monkeypatch):
    """A lost create race must reuse the winner's thread, not the parent channel.

    The gateway and the voice auto-join supervisor both ingest mentions; when
    the supervisor wins the thread create, the gateway used to fall back to the
    shared channel session, so the session and the reply both landed in the
    channel instead of the thread.
    """
    monkeypatch.setattr(fluxer_adapter, "_open_session_store", lambda home=None: Titles())
    adapter = make_adapter()
    adapter.create_thread = AsyncMock(side_effect=_RaceAlreadyCreated())
    adapter._request = AsyncMock(
        return_value={"id": "anchor", "thread": {"id": "race-thread", "name": "dog"}}
    )

    # Ingest-time path: the session must key on the adopted thread.
    thread_id = await adapter._start_thread_session_from_message("channel", "anchor", "hello")
    assert thread_id == "race-thread"
    assert adapter._threads_by_anchor[("channel", "anchor")] == "race-thread"

    # Send-time path adopts the same thread after a cache loss.
    adapter._threads_by_anchor.clear()
    assert await adapter._start_thread_from_message("channel", "anchor") == "race-thread"

    # Any other create failure still falls back to the channel.
    adapter.create_thread = AsyncMock(side_effect=RuntimeError("temporary rejection"))
    adapter._request = AsyncMock(return_value={"id": "anchor", "thread": None})
    assert await adapter._start_thread_session_from_message("channel", "anchor2", "hello") is None
    assert await adapter._start_thread_from_message("channel", "anchor2") is None


def test_temporary_reader_failure_recovers_after_bounded_retry(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(fluxer_adapter.time, "monotonic", lambda: clock[0])
    opened = []
    reader = Titles()

    def open_reader(home=None):
        opened.append(home)
        return None if len(opened) == 1 else reader

    monkeypatch.setattr(fluxer_adapter, "_open_session_store", open_reader)
    adapter = make_adapter()
    assert adapter._session_title_for_chat("channel", "alice") is None
    assert adapter._session_title_for_chat("channel", "alice") is None
    assert len(opened) == 1
    clock[0] += fluxer_adapter._TITLE_RENAME_POLL_INTERVAL + 1
    assert adapter._session_title_for_chat("channel", "alice") == "synthetic title"
    assert len(opened) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["😀" * 100, " \t\n "])
async def test_handoff_names_follow_shared_utf16_budget_and_empty_fallback(name):
    adapter = make_adapter()
    adapter.create_thread = AsyncMock(return_value={"id": "handoff-thread"})
    assert await adapter.create_handoff_thread("channel", name) == "handoff-thread"
    actual = adapter.create_thread.await_args.args[1]
    assert actual == (fluxer_adapter._collapse_thread_name(name) or fluxer_adapter._DEFAULT_THREAD_NAME)
    assert fluxer_adapter.utf16_len(actual) <= fluxer_adapter._THREAD_NAME_LIMIT
