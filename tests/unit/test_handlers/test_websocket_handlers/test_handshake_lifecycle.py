"""Tests for the websocket handshake lifecycle.

Covers:

* resources established before the handshake are cleaned up in reverse order when the upgrade is rejected,
  while dependencies that were never established are not invoked,
* a rejection response (``websocket.close``) is sent exactly once,
* resources stay alive until the connection is closed after a successful upgrade,
* handshake timeouts, client disconnects and dependency failures are each reported distinctly,
* per-request / per-connection dependency caches are not reused across handshakes.
"""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator
from typing import Any

import anyio

from litestar import Litestar, WebSocket, websocket
from litestar.di import NamedDependency, Provide
from litestar.exceptions import PermissionDeniedException, WebSocketException
from litestar.status_codes import WS_1002_PROTOCOL_ERROR, WS_1011_INTERNAL_ERROR, WS_1013_TRY_AGAIN_LATER
from litestar.testing import create_test_client


def _websocket_scope(path: str = "/") -> dict[str, Any]:
    return {
        "type": "websocket",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "ws",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }


async def _drive_app(app: Litestar, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drive ``app`` as an ASGI websocket connection, returning the messages it sent.

    Any exception escaping the app is swallowed: the tests assert on the ASGI messages the application produced,
    which fully describe the handshake outcome.
    """
    send_stream, receive_stream = anyio.create_memory_object_stream[dict[str, Any]](math.inf)
    sent: list[dict[str, Any]] = []
    for event in events:
        send_stream.send_nowait(event)

    async def receive() -> dict[str, Any]:
        return await receive_stream.receive()

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    try:
        await app(_websocket_scope(), receive, send)
    except BaseException:
        pass
    return sent


async def test_established_resources_cleaned_up_in_reverse_order_when_dependency_fails(
    anyio_backend: str,
) -> None:
    log: list[str] = []

    async def resource_a() -> AsyncGenerator[str, None]:
        log.append("a-start")
        try:
            yield "a"
        finally:
            log.append("a-cleanup")

    async def resource_b(a: NamedDependency[str]) -> AsyncGenerator[str, None]:
        log.append("b-start")
        raise RuntimeError("could not establish b")
        yield  # pragma: no cover

    @websocket(
        "/",
        dependencies={"a": Provide(resource_a), "b": Provide(resource_b)},
    )
    async def handler(socket: WebSocket, b: NamedDependency[str]) -> None:
        await socket.accept()

    app = Litestar(route_handlers=[handler])
    sent = await _drive_app(app, [{"type": "websocket.connect"}])
    # ``a`` was established and is torn down explicitly in reverse order; ``b`` never established
    assert log == ["a-start", "b-start", "a-cleanup"]
    # the failed dependency resolution is reported as an internal-error rejection, sent exactly once
    assert [(m["type"], m.get("code")) for m in sent] == [
        ("websocket.close", WS_1011_INTERNAL_ERROR),
    ]


async def test_unestablished_dependency_is_never_invoked(anyio_backend: str) -> None:
    log: list[str] = []

    async def resource_a() -> AsyncGenerator[str, None]:
        log.append("a-start")
        try:
            yield "a"
        finally:
            log.append("a-cleanup")

    async def resource_b(a: NamedDependency[str]) -> str:
        log.append("b-start")
        raise RuntimeError("could not establish b")

    async def resource_c(b: NamedDependency[str]) -> str:
        log.append("c-start")
        return "c"

    @websocket(
        "/",
        dependencies={
            "a": Provide(resource_a),
            "b": Provide(resource_b),
            "c": Provide(resource_c),
        },
    )
    async def handler(socket: WebSocket, c: NamedDependency[str]) -> None:
        await socket.accept()

    app = Litestar(route_handlers=[handler])
    sent = await _drive_app(app, [{"type": "websocket.connect"}])
    assert "c-start" not in log
    assert log == ["a-start", "b-start", "a-cleanup"]
    assert [m["type"] for m in sent] == ["websocket.close"]


async def test_guard_rejection_sends_a_single_close(anyio_backend: str) -> None:
    async def guard(connection: WebSocket, _: Any) -> None:
        raise PermissionDeniedException(detail="forbidden", status_code=403)

    @websocket("/", guards=[guard])
    async def handler(socket: WebSocket) -> None:
        await socket.accept()

    app = Litestar(route_handlers=[handler])
    sent = await _drive_app(app, [{"type": "websocket.connect"}])
    assert [(m["type"], m.get("code")) for m in sent] == [("websocket.close", 4500)]
    assert sent[0]["reason"] == "forbidden"


async def test_close_after_handler_already_closed_is_not_sent_twice(anyio_backend: str) -> None:
    @websocket("/")
    async def handler(socket: WebSocket) -> None:
        await socket.accept()
        await socket.receive_json()
        await socket.close()
        # an exception after the handler itself already closed the connection must not trigger a second close
        raise WebSocketException(detail="post close failure", code=4500)

    app = Litestar(route_handlers=[handler])
    sent = await _drive_app(
        app,
        [
            {"type": "websocket.connect"},
            {"type": "websocket.receive", "text": '{"hello": "world"}', "bytes": None},
        ],
    )
    assert [m["type"] for m in sent] == ["websocket.accept", "websocket.close"]


async def test_pre_accept_rejection_after_dependencies_cleans_up_in_reverse_order(
    anyio_backend: str,
) -> None:
    # dependencies (auth / database / temporary resources) are established before the upgrade, but the handler
    # may still reject the upgrade for protocol reasons before accepting; the rejection must be a single close and
    # the established resources must be torn down in reverse order.
    log: list[str] = []

    async def resource_a() -> AsyncGenerator[str, None]:
        log.append("a-start")
        try:
            yield "a"
        finally:
            log.append("a-cleanup")

    async def resource_b(a: NamedDependency[str]) -> AsyncGenerator[str, None]:
        log.append("b-start")
        try:
            yield "b"
        finally:
            log.append("b-cleanup")

    @websocket("/", dependencies={"a": Provide(resource_a), "b": Provide(resource_b)})
    async def handler(socket: WebSocket, a: NamedDependency[str], b: NamedDependency[str]) -> None:
        await socket.receive()
        # protocol condition not met -> reject without accepting
        await socket.close(code=WS_1002_PROTOCOL_ERROR)

    app = Litestar(route_handlers=[handler])
    sent = await _drive_app(app, [{"type": "websocket.connect"}])
    assert [(m["type"], m.get("code")) for m in sent] == [("websocket.close", WS_1002_PROTOCOL_ERROR)]
    assert log == ["a-start", "b-start", "b-cleanup", "a-cleanup"]


async def test_client_disconnect_before_accept_receives_no_response(anyio_backend: str) -> None:
    @websocket("/")
    async def handler(socket: WebSocket) -> None:
        await socket.accept()

    app = Litestar(route_handlers=[handler])
    sent = await _drive_app(app, [{"type": "websocket.disconnect", "code": 1001}])
    # the client is gone before the handshake completes: nothing can be rejected and no response is sent
    assert sent == []


async def test_handshake_timeout_is_reported_with_a_dedicated_single_close(anyio_backend: str) -> None:
    @websocket("/")
    async def handler(socket: WebSocket) -> None:
        await socket.accept(timeout=0.05)

    app = Litestar(route_handlers=[handler])
    # no ``websocket.connect`` event ever arrives
    sent = await _drive_app(app, [])
    assert [(m["type"], m.get("code")) for m in sent] == [
        ("websocket.close", WS_1013_TRY_AGAIN_LATER),
    ]


def test_resources_remain_alive_until_connection_closes_after_upgrade() -> None:
    log: list[str] = []

    async def resource() -> AsyncGenerator[str, None]:
        log.append("start")
        try:
            yield "resource"
        finally:
            log.append("cleanup")

    @websocket("/", dependencies={"resource": Provide(resource)})
    async def handler(socket: WebSocket, resource: NamedDependency[str]) -> None:
        await socket.accept()
        await socket.receive_json()
        # the resource must still be alive while the connection is open
        assert log == ["start"]
        assert resource == "resource"
        await socket.close()
        # it is only cleaned up once the handler / connection has finished
        assert log == ["start"]

    with create_test_client(handler) as client, client.websocket_connect("/") as ws:
        ws.send_json({"ping": "pong"})

    assert log == ["start", "cleanup"]


def test_cached_dependency_is_not_reused_across_handshakes() -> None:
    calls = 0

    async def cached_dependency() -> int:
        nonlocal calls
        calls += 1
        return calls

    @websocket("/", dependencies={"value": Provide(cached_dependency, use_cache=True)})
    async def handler(socket: WebSocket, value: NamedDependency[int]) -> None:
        await socket.accept()
        await socket.send_json({"value": value})
        await socket.close()

    with create_test_client(handler) as client:
        with client.websocket_connect("/") as ws:
            assert ws.receive_json() == {"value": 1}
        with client.websocket_connect("/") as ws:
            assert ws.receive_json() == {"value": 2}

    assert calls == 2


def test_cached_dependency_resolved_once_per_connection() -> None:
    calls = 0

    async def cached_dependency() -> int:
        nonlocal calls
        calls += 1
        return calls

    async def dependent(value: NamedDependency[int]) -> int:
        return value + 10

    @websocket(
        "/",
        dependencies={
            "value": Provide(cached_dependency, use_cache=True),
            "dependent": Provide(dependent),
        },
    )
    async def handler(socket: WebSocket, value: NamedDependency[int], dependent: NamedDependency[int]) -> None:
        await socket.accept()
        await socket.send_json({"value": value, "dependent": dependent})
        await socket.close()

    with create_test_client(handler) as client:
        with client.websocket_connect("/") as ws:
            assert ws.receive_json() == {"value": 1, "dependent": 11}
        with client.websocket_connect("/") as ws:
            assert ws.receive_json() == {"value": 2, "dependent": 12}

    assert calls == 2


async def test_established_resources_in_concurrent_batch_are_all_cleaned(anyio_backend: str) -> None:
    # resources within a single batch are established concurrently, so their establishment order is not defined;
    # every resource that did establish must nevertheless be cleaned up (in reverse completion order), while the
    # failing one that never established is not.
    log: list[str] = []

    async def first() -> AsyncGenerator[str, None]:
        try:
            yield "first"
        finally:
            log.append("first-cleanup")

    async def second() -> AsyncGenerator[str, None]:
        try:
            yield "second"
        finally:
            log.append("second-cleanup")

    async def failing() -> AsyncGenerator[str, None]:
        log.append("failing-start")
        raise RuntimeError("boom")
        yield  # pragma: no cover

    @websocket(
        "/",
        dependencies={
            "first": Provide(first),
            "second": Provide(second),
            "failing": Provide(failing),
        },
    )
    async def handler(
        socket: WebSocket,
        first: NamedDependency[str],
        second: NamedDependency[str],
        failing: NamedDependency[str],
    ) -> None:
        await socket.accept()

    app = Litestar(route_handlers=[handler])
    sent = await _drive_app(app, [{"type": "websocket.connect"}])
    assert [m["type"] for m in sent] == ["websocket.close"]
    cleanups = [entry for entry in log if entry.endswith("cleanup")]
    assert set(cleanups) == {"first-cleanup", "second-cleanup"}
    assert "failing-cleanup" not in log
