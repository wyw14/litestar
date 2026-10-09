import logging
from typing import Any

import anyio
import pytest

from litestar import Litestar, get
from litestar.background_tasks import (
    BackgroundTask,
    BackgroundTaskEndReason,
    BackgroundTaskManager,
    BackgroundTasks,
    BackgroundTaskStatus,
)
from litestar.enums import HttpMethod, ScopeType
from litestar.status_codes import HTTP_200_OK
from litestar.testing import RequestFactory, create_test_client
from litestar.utils.scope.state import ScopeState


async def test_background_tasks_regular_execution() -> None:
    values: list[int] = []

    def extend_values(values_to_extend: list[int]) -> None:
        values.extend(values_to_extend)

    tasks = BackgroundTasks(
        [BackgroundTask(extend_values, [1, 2, 3]), BackgroundTask(extend_values, values_to_extend=[4, 5, 6])]
    )

    @get("/", background=tasks)
    def handler() -> None:
        return None

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert values == [1, 2, 3, 4, 5, 6]


async def test_background_tasks_task_group_execution() -> None:
    values: list[int] = []

    def extend_values(values_to_extend: list[int]) -> None:
        values.extend(values_to_extend)

    tasks = BackgroundTasks(
        [BackgroundTask(extend_values, [1, 2, 3]), BackgroundTask(extend_values, values_to_extend=[4, 5, 6])],
        run_in_task_group=True,
    )

    @get("/", background=tasks)
    def handler() -> None:
        return None

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert set(values) == {1, 2, 3, 4, 5, 6}


async def test_background_task_exception_goes_to_after_exception_without_touching_response() -> None:
    reported: list[tuple[Exception, Any]] = []

    async def after_exception(exc: Exception, scope: Any) -> None:
        reported.append((exc, scope))

    async def failing_task() -> None:
        raise ValueError("boom")

    @get("/", background=BackgroundTask(failing_task))
    async def handler() -> dict[str, str]:
        return {"hello": "world"}

    with create_test_client(handler, after_exception=[after_exception]) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.json() == {"hello": "world"}

    assert len(reported) == 1
    exc, scope = reported[0]
    assert isinstance(exc, ValueError)
    assert str(exc) == "boom"
    assert scope["type"] == ScopeType.HTTP

    manager = ScopeState.from_scope(scope).background_task_manager
    assert isinstance(manager, BackgroundTaskManager)
    assert manager.is_finished
    assert manager.task_statuses == (BackgroundTaskStatus.FAILED,)
    assert manager.end_reason is BackgroundTaskEndReason.COMPLETED


async def test_background_tasks_report_once_and_continue_cleanup_in_registration_order() -> None:
    values: list[int] = []
    hook_calls: list[Exception] = []
    managers: list[BackgroundTaskManager] = []

    async def after_exception(exc: Exception, scope: Any) -> None:
        hook_calls.append(exc)
        managers.append(ScopeState.from_scope(scope).background_task_manager)

    async def add_value(value: int) -> None:
        values.append(value)

    async def fail(value: int) -> None:
        raise RuntimeError(f"failure-{value}")

    tasks = BackgroundTasks(
        [
            BackgroundTask(add_value, 1),
            BackgroundTask(fail, 2),
            BackgroundTask(add_value, 3),
            BackgroundTask(fail, 4),
        ]
    )

    @get("/", background=tasks)
    async def handler() -> None:
        return None

    with create_test_client(handler, after_exception=[after_exception]) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK

    # tasks not blocked by a failure, and run / report in registration order
    assert values == [1, 3]
    assert [str(e) for e in hook_calls] == ["failure-2", "failure-4"]

    manager = managers[0]
    assert manager.task_statuses == (
        BackgroundTaskStatus.SUCCEEDED,
        BackgroundTaskStatus.FAILED,
        BackgroundTaskStatus.SUCCEEDED,
        BackgroundTaskStatus.FAILED,
    )
    assert [type(e) for e in manager.task_exceptions] == [type(None), RuntimeError, type(None), RuntimeError]
    assert manager.end_reason is BackgroundTaskEndReason.COMPLETED


async def test_background_tasks_task_group_reports_each_failure_once() -> None:
    hook_calls: list[Exception] = []

    async def after_exception(exc: Exception, scope: Any) -> None:
        hook_calls.append(exc)

    async def fail(value: int) -> None:
        raise RuntimeError(f"failure-{value}")

    tasks = BackgroundTasks(
        [BackgroundTask(fail, 1), BackgroundTask(fail, 2)],
        run_in_task_group=True,
    )

    @get("/", background=tasks)
    async def handler() -> None:
        return None

    with create_test_client(handler, after_exception=[after_exception]) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK

    assert sorted(str(e) for e in hook_calls) == ["failure-1", "failure-2"]
    assert len(hook_calls) == 2


async def test_a_failing_after_exception_hook_does_not_break_cleanup(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[str] = []

    async def bad_hook(exc: Exception, scope: Any) -> None:
        calls.append("bad")
        raise RuntimeError("hook failure")

    async def good_hook(exc: Exception, scope: Any) -> None:
        calls.append("good")

    async def task_ok() -> None:
        calls.append("task")

    async def task_fail() -> None:
        raise ValueError("boom")

    @get("/", background=BackgroundTasks([BackgroundTask(task_ok), BackgroundTask(task_fail)]))
    async def handler() -> None:
        return None

    with caplog.at_level(logging.ERROR):
        with create_test_client(handler, after_exception=[bad_hook, good_hook]) as client:
            response = client.get("/")
            assert response.status_code == HTTP_200_OK

    assert calls == ["task", "bad", "good"]
    assert "after_exception" in caplog.text


async def test_new_request_does_not_inherit_previous_request_background_task_state() -> None:
    values: list[int] = []
    hook_calls: list[Exception] = []
    per_request_managers: list[BackgroundTaskManager] = []

    async def after_exception(exc: Exception, scope: Any) -> None:
        hook_calls.append(exc)
        per_request_managers.append(ScopeState.from_scope(scope).background_task_manager)

    async def add_value() -> None:
        values.append(1)

    async def fail() -> None:
        raise RuntimeError("boom")

    # the same container instance is shared across requests via the handler
    shared_tasks = BackgroundTasks([BackgroundTask(add_value), BackgroundTask(fail)])

    @get("/", background=shared_tasks)
    async def handler() -> None:
        return None

    with create_test_client(handler, after_exception=[after_exception]) as client:
        first = client.get("/")
        second = client.get("/")
        assert first.status_code == HTTP_200_OK
        assert second.status_code == HTTP_200_OK

    # the tasks ran independently for both requests
    assert values == [1, 1]
    # one failure reported per request, not accumulated on a shared manager
    assert len(hook_calls) == 2
    assert per_request_managers[0] is not per_request_managers[1]
    for manager in per_request_managers:
        assert manager.task_statuses == (BackgroundTaskStatus.SUCCEEDED, BackgroundTaskStatus.FAILED)
        assert manager.is_finished


class _FakeApp:
    def __init__(self, *, shutdown_event: anyio.Event | None) -> None:
        self.after_exception: list = []
        self.managers: set[BackgroundTaskManager] = set()
        self._shutdown_event = shutdown_event

    def register_background_task_manager(self, manager: BackgroundTaskManager) -> None:
        self.managers.add(manager)

    def unregister_background_task_manager(self, manager: BackgroundTaskManager) -> None:
        self.managers.discard(manager)

    def get_background_tasks_shutdown_event(self) -> anyio.Event | None:
        return self._shutdown_event


async def test_manager_completed_reason_with_blocking_receive() -> None:
    app = _FakeApp(shutdown_event=None)
    manager = BackgroundTaskManager()
    ran = False

    async def task() -> None:
        nonlocal ran
        await anyio.sleep(0)
        ran = True

    manager.add_task(task)

    async def receive() -> dict[str, str]:
        # blocks forever: no disconnect arrives while the tasks run
        event = anyio.Event()
        await event.wait()
        return {"type": "http.disconnect"}  # pragma: no cover

    await manager({"litestar_app": app}, receive)  # type: ignore[arg-type]

    assert ran
    assert manager.end_reason is BackgroundTaskEndReason.COMPLETED
    assert manager.task_statuses == (BackgroundTaskStatus.SUCCEEDED,)
    assert app.managers == set()


async def test_manager_request_cancelled_reason() -> None:
    app = _FakeApp(shutdown_event=None)
    manager = BackgroundTaskManager()
    task_started = anyio.Event()
    disconnect = anyio.Event()
    release = anyio.Event()
    call_count = 0

    async def long_task() -> None:
        task_started.set()
        await release.wait()

    async def pending_task() -> None:
        pass

    manager.add_task(long_task)
    manager.add_task(pending_task)  # pending task must be cancelled without running

    async def receive() -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # the request event is already pending when the phase starts
            return {"type": "http.request", "body": b"", "more_body": False}
        # a genuine disconnect that arrives while tasks are running
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async with anyio.create_task_group() as task_group:

        async def trigger_disconnect() -> None:
            await task_started.wait()
            disconnect.set()

        task_group.start_soon(trigger_disconnect)
        await manager({"litestar_app": app}, receive)  # type: ignore[arg-type]

    assert manager.end_reason is BackgroundTaskEndReason.REQUEST_CANCELLED
    assert manager.task_statuses == (BackgroundTaskStatus.CANCELLED, BackgroundTaskStatus.CANCELLED)
    statuses = manager._entries  # type: ignore[attr-defined]
    assert all(entry.cancel_reason is BackgroundTaskEndReason.REQUEST_CANCELLED for entry in statuses)


async def test_manager_app_shutdown_reason() -> None:
    shutdown_event = anyio.Event()
    app = _FakeApp(shutdown_event=shutdown_event)
    manager = BackgroundTaskManager()
    task_started = anyio.Event()
    release = anyio.Event()

    async def long_task() -> None:
        task_started.set()
        await release.wait()

    manager.add_task(long_task)

    async def receive() -> dict[str, str]:
        event = anyio.Event()
        await event.wait()
        return {"type": "http.disconnect"}  # pragma: no cover

    async with anyio.create_task_group() as task_group:

        async def trigger_shutdown() -> None:
            await task_started.wait()
            shutdown_event.set()

        task_group.start_soon(trigger_shutdown)
        await manager({"litestar_app": app}, receive)  # type: ignore[arg-type]

    assert manager.end_reason is BackgroundTaskEndReason.APP_SHUTDOWN
    assert manager.task_statuses == (BackgroundTaskStatus.CANCELLED,)
    assert manager._entries[0].cancel_reason is BackgroundTaskEndReason.APP_SHUTDOWN  # type: ignore[attr-defined]


async def test_manager_external_cancellation_keeps_distinct_reasons() -> None:
    async def run(external_shutdown_event: anyio.Event | None, expected: BackgroundTaskEndReason) -> None:
        app = _FakeApp(shutdown_event=external_shutdown_event)
        manager = BackgroundTaskManager()
        started = anyio.Event()
        release = anyio.Event()

        async def long_task() -> None:
            started.set()
            await release.wait()

        manager.add_task(long_task)

        async def receive() -> dict[str, str]:
            event = anyio.Event()
            await event.wait()
            return {"type": "http.disconnect"}  # pragma: no cover

        cancel_scope = anyio.CancelScope()

        async with anyio.create_task_group() as task_group:

            async def cancel() -> None:
                await started.wait()
                if external_shutdown_event is not None:
                    external_shutdown_event.set()
                await anyio.sleep(0)
                cancel_scope.cancel()

            task_group.start_soon(cancel)

            async def run_manager() -> None:
                # the dedicated scope absorbs the outer cancellation after the
                # manager itself has recorded the distinct reason
                with cancel_scope:
                    await manager({"litestar_app": app}, receive)  # type: ignore[arg-type]

            task_group.start_soon(run_manager)

        assert manager.end_reason is expected

    await run(None, BackgroundTaskEndReason.REQUEST_CANCELLED)
    await run(anyio.Event(), BackgroundTaskEndReason.APP_SHUTDOWN)


async def test_request_cancelled_during_background_tasks_keeps_sent_response() -> None:
    task_started = anyio.Event()
    disconnect_gate = anyio.Event()
    release = anyio.Event()
    response_done = anyio.Event()
    sent: dict[str, Any] = {"status": None, "final_body": None}

    async def long_task() -> None:
        task_started.set()
        await release.wait()

    @get("/", background=BackgroundTask(long_task))
    async def handler() -> dict[str, str]:
        return {"hello": "world"}

    app = Litestar(route_handlers=[handler])
    scope = RequestFactory(app=app)._create_scope(path="/", http_method=HttpMethod.GET)
    call_count = 0

    async def receive() -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect_gate.wait()
        return {"type": "http.disconnect"}

    async def send(message: Any) -> None:
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body" and not message.get("more_body"):
            sent["final_body"] = message["body"]
            response_done.set()

    async with anyio.create_task_group() as task_group:

        async def trigger_disconnect() -> None:
            await response_done.wait()
            await task_started.wait()
            disconnect_gate.set()

        task_group.start_soon(trigger_disconnect)
        await app(scope, receive, send)  # type: ignore[arg-type]

    assert sent["status"] == HTTP_200_OK
    assert sent["final_body"] == b'{"hello":"world"}'
    manager = ScopeState.from_scope(scope).background_task_manager
    assert manager.end_reason is BackgroundTaskEndReason.REQUEST_CANCELLED
    assert manager.task_statuses == (BackgroundTaskStatus.CANCELLED,)


async def test_app_shutdown_cancels_in_flight_background_tasks_with_shutdown_reason() -> None:
    import asyncio

    task_started = anyio.Event()
    release = anyio.Event()
    response_done = anyio.Event()

    async def long_task() -> None:
        task_started.set()
        await release.wait()

    @get("/", background=BackgroundTask(long_task))
    async def handler() -> dict[str, str]:
        return {"hello": "world"}

    app = Litestar(route_handlers=[handler])
    scope = RequestFactory(app=app)._create_scope(path="/", http_method=HttpMethod.GET)
    call_count = 0

    async def receive() -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        # the client stays connected: no disconnect ever arrives
        gate = anyio.Event()
        await gate.wait()
        return {"type": "http.disconnect"}  # pragma: no cover

    async def send(message: Any) -> None:
        if message["type"] == "http.response.body" and not message.get("more_body"):
            response_done.set()

    async with app.lifespan():
        request_future = asyncio.ensure_future(app(scope, receive, send))  # type: ignore[arg-type]
        await response_done.wait()
        await task_started.wait()
        # exiting the lifespan shuts the application down: in-flight background
        # tasks must be cancelled with the APP_SHUTDOWN reason and awaited
    await request_future

    manager = ScopeState.from_scope(scope).background_task_manager
    assert manager.end_reason is BackgroundTaskEndReason.APP_SHUTDOWN
    assert manager.task_statuses == (BackgroundTaskStatus.CANCELLED,)
    assert manager._entries[0].cancel_reason is BackgroundTaskEndReason.APP_SHUTDOWN  # type: ignore[attr-defined]
    assert app._background_task_managers == set()


async def test_tasks_can_be_added_from_a_running_task_in_registration_order() -> None:
    app = _FakeApp(shutdown_event=None)
    manager = BackgroundTaskManager()
    values: list[int] = []

    async def first() -> None:
        values.append(1)
        # schedule another task while the response has already been sent
        manager.add_task(third)

    async def second() -> None:
        values.append(2)

    async def third() -> None:
        values.append(3)

    manager.add_task(first)
    manager.add_task(second)

    await manager({"litestar_app": app}, _never_returns)  # type: ignore[arg-type]

    assert values == [1, 2, 3]
    assert manager.end_reason is BackgroundTaskEndReason.COMPLETED
    assert manager.task_statuses == (
        BackgroundTaskStatus.SUCCEEDED,
        BackgroundTaskStatus.SUCCEEDED,
        BackgroundTaskStatus.SUCCEEDED,
    )


async def test_cannot_add_tasks_while_running_concurrently_or_after_finish() -> None:
    app = _FakeApp(shutdown_event=None)
    release = anyio.Event()
    started = anyio.Event()

    async def blocker() -> None:
        started.set()
        await release.wait()

    async def noop() -> None:
        return None

    manager = BackgroundTaskManager(run_in_task_group=True)
    manager.add_task(blocker)

    async with anyio.create_task_group() as task_group:

        async def run_manager() -> None:
            await manager({"litestar_app": app}, _never_returns)  # type: ignore[arg-type]

        task_group.start_soon(run_manager)
        await started.wait()
        with pytest.raises(RuntimeError):
            manager.add_task(noop)
        release.set()

    with pytest.raises(RuntimeError):
        manager.add_task(noop)


async def test_cannot_add_tasks_to_a_finished_manager() -> None:
    async def noop() -> None:
        return None

    app = _FakeApp(shutdown_event=None)
    manager = BackgroundTaskManager()
    manager.add_task(noop)
    await manager({"litestar_app": app}, _never_returns)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        manager.add_task(noop)


async def _never_returns() -> dict[str, str]:
    gate = anyio.Event()
    await gate.wait()
    return {"type": "http.disconnect"}  # type: ignore[return-value]  # pragma: no cover
