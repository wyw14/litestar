from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, final

from anyio import CancelScope, Event, create_task_group, get_cancelled_exc_class, sleep
from typing_extensions import ParamSpec

from litestar.utils.sync import ensure_async_callable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from litestar.types import Receive, Scope

__all__ = (
    "BackgroundTask",
    "BackgroundTaskEndReason",
    "BackgroundTaskManager",
    "BackgroundTaskStatus",
    "BackgroundTasks",
)

P = ParamSpec("P")

logger = logging.getLogger(__name__)


class BackgroundTaskStatus(StrEnum):
    """Lifecycle status of a single background task within a request."""

    PENDING = "pending"
    """The task has been registered but has not started yet."""
    RUNNING = "running"
    """The task is currently executing."""
    SUCCEEDED = "succeeded"
    """The task completed without raising an exception."""
    FAILED = "failed"
    """The task raised an exception, which was reported to the application error channel."""
    CANCELLED = "cancelled"
    """The task was cancelled (request disconnect or application shutdown) before it could finish."""


class BackgroundTaskEndReason(StrEnum):
    """The reason a request's background task processing ended."""

    REQUEST_CANCELLED = "request_cancelled"
    """The client disconnected while background tasks were still running."""
    COMPLETED = "completed"
    """All background tasks finished normally."""
    APP_SHUTDOWN = "app_shutdown"
    """The application is shutting down and cancelled the still running background tasks."""


class BackgroundTask:
    """A container for a 'background' task function.

    Background tasks are called once a Response finishes.
    """

    __slots__ = ("args", "fn", "kwargs")

    def __init__(self, fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> None:
        """Initialize ``BackgroundTask``.

        Args:
            fn: A sync or async function to call as the background task.
            *args: Args to pass to the func.
            **kwargs: Kwargs to pass to the func
        """
        self.fn = ensure_async_callable(fn)
        self.args = args
        self.kwargs = kwargs

    async def __call__(self) -> None:
        """Call the wrapped function with the passed in arguments.

        Returns:
            None
        """
        await self.fn(*self.args, **self.kwargs)


class BackgroundTasks:
    """A container for multiple 'background' task functions.

    Background tasks are called once a Response finishes.
    """

    __slots__ = ("run_in_task_group", "tasks")

    def __init__(self, tasks: Iterable[BackgroundTask], run_in_task_group: bool = False) -> None:
        """Initialize ``BackgroundTasks``.

        Args:
            tasks: An iterable of :class:`BackgroundTask <.background_tasks.BackgroundTask>` instances.
            run_in_task_group: If you set this value to ``True`` than the tasks will run concurrently, using
                a :class:`TaskGroup <anyio.abc.TaskGroup>`. Note: This will not preserve execution order.
        """
        self.tasks = list(tasks)
        self.run_in_task_group = run_in_task_group

    async def __call__(self) -> None:
        """Call the wrapped background tasks.

        Returns:
            None
        """
        if self.run_in_task_group:
            async with create_task_group() as task_group:
                for task in self.tasks:
                    task_group.start_soon(task)
        else:
            for task in self.tasks:
                await task()


@final
class _TaskEntry:
    __slots__ = ("cancel_reason", "cancel_scope", "exception", "status", "task")

    def __init__(self, task: BackgroundTask) -> None:
        self.task = task
        self.status: BackgroundTaskStatus = BackgroundTaskStatus.PENDING
        self.exception: BaseException | None = None
        self.cancel_reason: BackgroundTaskEndReason | None = None
        # a dedicated cancel scope lets the manager cancel an individual task
        # while distinguishing *why* it was cancelled
        self.cancel_scope = CancelScope()


class BackgroundTaskManager:
    """Orchestrate the background tasks registered for a single request.

    A new manager is created for every request and stored in the connection scope,
    so requests never share background task state. It is responsible for:

        - running the registered tasks after the response has been sent,
        - keeping tasks in a registration ordered lifecycle,
        - routing a task's exception into the application's unified error
          channel (the ``after_exception`` hooks) exactly once, without
          modifying the already completed response,
        - distinguishing between request cancellation, normal completion and
          application shutdown as distinct end reasons.
    """

    __slots__ = (
        "_end_reason",
        "_entries",
        "_finished",
        "_finished_event",
        "_remaining",
        "_run_in_task_group",
        "_running",
        "_scope",
        "_shutdown_event",
        "_stopping",
    )

    def __init__(self, *, run_in_task_group: bool = False) -> None:
        """Initialize ``BackgroundTaskManager``.

        Args:
            run_in_task_group: If ``True``, tasks run concurrently in a task group
                (execution order is not preserved). If ``False`` (the default), tasks
                run sequentially, in registration order.
        """
        self._entries: list[_TaskEntry] = []
        self._run_in_task_group = run_in_task_group
        self._remaining = 0
        self._running = False
        self._finished = False
        self._stopping = False
        self._end_reason: BackgroundTaskEndReason | None = None
        self._finished_event = Event()
        self._scope: Scope | None = None
        self._shutdown_event: Event | None = None

    @property
    def end_reason(self) -> BackgroundTaskEndReason | None:
        """The reason the background tasks ended, or ``None`` while they are still running."""
        return self._end_reason

    @property
    def is_finished(self) -> bool:
        """Whether all background tasks have run (or been cleaned up) and the manager has closed."""
        return self._finished

    @property
    def task_statuses(self) -> tuple[BackgroundTaskStatus, ...]:
        """The lifecycle status of every registered task, in registration order."""
        return tuple(entry.status for entry in self._entries)

    @property
    def task_exceptions(self) -> tuple[BaseException | None, ...]:
        """The exception raised by every registered task, in registration order.

        A value of ``None`` means the task did not fail.
        """
        return tuple(entry.exception for entry in self._entries)

    @classmethod
    def from_scope(cls, scope: Scope) -> BackgroundTaskManager:
        """Return the manager bound to the given ASGI scope, creating it on first access.

        The manager is stored in the per-connection scope state, which guarantees that
        a new request never inherits a previous request's background task state.

        Args:
            scope: The ASGI connection scope.

        Returns:
            The :class:`BackgroundTaskManager` instance for the request.
        """
        from litestar.types import Empty
        from litestar.utils.scope.state import ScopeState

        state = ScopeState.from_scope(scope)
        manager = state.background_task_manager
        if manager is Empty:
            manager = cls()
            state.background_task_manager = manager
        return manager

    def add(self, background: BackgroundTask | BackgroundTasks) -> None:
        """Register a background task or a collection of background tasks, in order.

        Args:
            background: A :class:`BackgroundTask` or :class:`BackgroundTasks` instance.

        Raises:
            RuntimeError: If tasks are added after the manager has already finished, or if
                tasks are added while the manager is already running tasks concurrently.
        """
        if isinstance(background, BackgroundTasks):
            self._run_in_task_group = self._run_in_task_group or background.run_in_task_group
            for task in background.tasks:
                self._add_entry(task)
        else:
            self._add_entry(background)

    def add_task(self, fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> None:
        """Register a single callable as a background task, in order.

        Args:
            fn: A sync or async function to call as the background task.
            *args: Args to pass to the function.
            **kwargs: Kwargs to pass to the function.
        """
        self._add_entry(BackgroundTask(fn, *args, **kwargs))

    def _add_entry(self, task: BackgroundTask) -> None:
        if self._finished:
            raise RuntimeError("Cannot add background tasks after the background task manager has finished")
        if self._running and self._run_in_task_group:
            # concurrent tasks are scheduled up front, so late additions would never run
            raise RuntimeError("Cannot add background tasks while tasks are already running concurrently")
        self._entries.append(_TaskEntry(task))
        self._remaining += 1

    async def wait_closed(self) -> None:
        """Wait until the manager has finished running and cleaning up all its tasks."""
        await self._finished_event.wait()

    async def __call__(self, scope: Scope, receive: Receive) -> None:
        """Run the registered background tasks for a request.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive function, used to listen for client disconnects.

        Returns:
            None
        """
        if self._finished:
            return
        self._scope = scope
        app = scope["litestar_app"]
        self._shutdown_event = app.get_background_tasks_shutdown_event()
        app.register_background_task_manager(self)
        try:
            if self._entries:
                self._running = True
                await self._run(scope=scope, receive=receive)
            else:
                self._end_reason = BackgroundTaskEndReason.COMPLETED
        except get_cancelled_exc_class():
            # the request task itself was cancelled from outside: either the
            # application is shutting down (the lifespan event was set) or the
            # request was cancelled by the server / client
            if self._end_reason is None:
                if self._shutdown_event is not None and self._shutdown_event.is_set():
                    self._end_reason = BackgroundTaskEndReason.APP_SHUTDOWN
                else:
                    self._end_reason = BackgroundTaskEndReason.REQUEST_CANCELLED
            raise
        finally:
            self._running = False
            self._cleanup()
            app.unregister_background_task_manager(self)
            self._finished = True
            self._finished_event.set()

    async def _run(self, scope: Scope, receive: Receive) -> None:
        all_done = Event()
        try:
            async with create_task_group() as task_group:
                if self._run_in_task_group:
                    for index in range(len(self._entries)):
                        task_group.start_soon(self._run_entry, index, scope, all_done)
                else:
                    task_group.start_soon(self._run_entries_sequentially, scope, all_done)
                task_group.start_soon(self._watch_request_cancellation, receive)
                if self._shutdown_event is not None:
                    task_group.start_soon(self._watch_app_shutdown)

                await all_done.wait()
                # every task is accounted for: mark a normal completion and cancel
                # the watcher tasks, whose CancelledError is swallowed by the
                # task group's cancel scope
                if self._end_reason is None:
                    self._end_reason = BackgroundTaskEndReason.COMPLETED
                task_group.cancel_scope.cancel()
        except get_cancelled_exc_class():
            self._mark_external_cancellation_reason()
            raise

    async def _run_entries_sequentially(self, scope: Scope, all_done: Event) -> None:
        # the while loop (instead of iterating a fixed list) also picks up
        # tasks appended while the sequential run is in progress
        index = 0
        while index < len(self._entries):
            if self._stopping:
                for tail_index in range(index, len(self._entries)):
                    self._cancel_pending_entry(self._entries[tail_index])
                    self._mark_done(all_done)
                return
            await self._run_entry(index, scope, all_done)
            index += 1

    async def _run_entry(self, index: int, scope: Scope, all_done: Event) -> None:
        entry = self._entries[index]
        entry.status = BackgroundTaskStatus.RUNNING
        failure: Exception | None = None
        try:
            # a cancellation of this scope (request disconnect / app shutdown)
            # makes the ``with`` block absorb the resulting CancelledError,
            # whereas an outer cancellation propagates normally
            with entry.cancel_scope:
                await entry.task()
        except Exception as exc:  # noqa: BLE001
            # route the failure into the application's unified error channel,
            # exactly once, and never let it escape into the (already sent)
            # response
            entry.status = BackgroundTaskStatus.FAILED
            entry.exception = exc
            failure = exc
        else:
            if entry.cancel_scope.cancel_called:
                entry.status = BackgroundTaskStatus.CANCELLED
                entry.cancel_reason = self._end_reason
            else:
                entry.status = BackgroundTaskStatus.SUCCEEDED

        if failure is not None:
            await self._report_failure(failure, scope)
        self._mark_done(all_done)

    async def _watch_request_cancellation(self, receive: Receive) -> None:
        # Only a disconnect observed *while* the tasks are executing is a
        # request cancellation. A disconnect that is already surfaced as soon
        # as the response phase ends (the ASGI test client and some servers
        # expose connection teardown this way) is normal teardown and must not
        # cancel tasks that still belong to the completed request.
        while True:
            message, was_pending = await _probe_next_receive_event(receive)
            if message["type"] != "http.disconnect":
                # drain a late ``http.request`` event and keep watching
                continue
            if not was_pending:
                self._cancel(BackgroundTaskEndReason.REQUEST_CANCELLED)
            return

    async def _watch_app_shutdown(self) -> None:
        if self._shutdown_event is None:
            return
        await self._shutdown_event.wait()
        self._cancel(BackgroundTaskEndReason.APP_SHUTDOWN)

    def _mark_done(self, all_done: Event) -> None:
        self._remaining -= 1
        if self._remaining == 0:
            all_done.set()

    def _cancel_pending_entry(self, entry: _TaskEntry) -> None:
        if entry.status is BackgroundTaskStatus.PENDING:
            entry.status = BackgroundTaskStatus.CANCELLED
            entry.cancel_reason = self._end_reason

    def _mark_external_cancellation_reason(self) -> None:
        if self._end_reason is not None:
            return
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            self._end_reason = BackgroundTaskEndReason.APP_SHUTDOWN
        else:
            self._end_reason = BackgroundTaskEndReason.REQUEST_CANCELLED

    def _cancel(self, reason: BackgroundTaskEndReason) -> None:
        # the first trigger (request disconnect vs app shutdown) keeps its reason
        if self._end_reason is None:
            self._end_reason = reason
        self._stopping = True
        for entry in self._entries:
            if not entry.cancel_scope.cancel_called:
                entry.cancel_scope.cancel()

    async def _report_failure(self, exc: Exception, scope: Scope) -> None:
        app = scope["litestar_app"]
        for hook in app.after_exception:
            try:
                await hook(exc, scope)
            except Exception:
                # a failing error hook must not stop other hooks or the cleanup
                logger.exception(
                    "Unhandled exception in an `after_exception` hook while handling a background task exception"
                )

    def _cleanup(self) -> None:
        # finalize entries in registration order so cleanup order is deterministic
        for entry in self._entries:
            if entry.status in (BackgroundTaskStatus.PENDING, BackgroundTaskStatus.RUNNING):
                entry.status = BackgroundTaskStatus.CANCELLED
                entry.cancel_reason = self._end_reason or BackgroundTaskEndReason.APP_SHUTDOWN


async def _probe_next_receive_event(receive: Receive) -> tuple[dict[str, Any], bool]:
    """Return the next ASGI receive event and whether it was already pending.

    The receiver gets one scheduler turn: if it returns without waiting, the
    event was already queued when the response phase ended (normal connection
    teardown). If it parked waiting, the event genuinely arrived while
    background tasks were running.
    """
    delivered = Event()
    result: dict[str, Any] = {}

    async def receive_once() -> None:
        result["message"] = await receive()
        delivered.set()

    async with create_task_group() as probe_group:
        probe_group.start_soon(receive_once)
        # one scheduler turn for the receiver to return a pending event or park
        await sleep(0)
        was_pending = delivered.is_set()
        if not was_pending:
            await delivered.wait()

    return result["message"], was_pending
