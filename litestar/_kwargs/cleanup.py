from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from inspect import isasyncgen
from typing import TYPE_CHECKING, Any

from litestar.utils import ensure_async_callable

__all__ = ("DependencyCleanupGroup",)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
    from types import TracebackType

    from litestar.types import AnyGenerator


class DependencyCleanupGroup(AbstractAsyncContextManager):
    """Wrapper for generator based dependencies.

    Simplify cleanup by wrapping :func:`next` / :func:`anext` calls and providing facilities to
    :meth:`throw <generator.throw>` / :meth:`athrow <agen.athrow>` into all generators consecutively. An instance of
    this class can be used as a contextmanager, which will automatically throw any exceptions into its generators. All
    exceptions caught in this manner will be re-raised after they have been thrown in the generators.
    """

    def __init__(self, generators: list[AnyGenerator] | None = None) -> None:
        """Initialize ``DependencyCleanupGroup``.

        Args:
            generators: An optional list of generators to be called at cleanup
        """
        self._generators = generators or []
        self._closed = False

    def add(self, generator: Generator[Any, None, None] | AsyncGenerator[Any, None]) -> None:
        """Add a new generator to the group.

        Args:
            generator: The generator to add

        Returns:
            None
        """
        if self._closed:
            raise RuntimeError("Cannot call .add on a closed DependencyCleanupGroup")
        self._generators.append(generator)

    @staticmethod
    def _wrap_next(generator: AnyGenerator) -> Callable[[], Awaitable[None]]:
        if isasyncgen(generator):

            async def wrapped_async() -> None:
                await anext(generator, None)

            return wrapped_async

        def wrapped() -> None:
            next(generator, None)  # type: ignore[arg-type]

        return ensure_async_callable(wrapped)

    async def close(self, exc: BaseException | None = None) -> None:
        if self._closed:
            raise RuntimeError("Cannot call cleanup on a closed DependencyCleanupGroup")

        self._closed = True

        if exc is None:
            await self._cleanup()
        else:
            await self._throw(exc)

    async def _cleanup(self) -> None:
        """Execute cleanup by calling :func:`next` / :func:`anext` on all generators.

        Generators are cleaned up sequentially in reverse order of their creation, i.e. the most recently
        established resource is torn down first (LIFO), mirroring nested context manager semantics. If a generator
        raises during cleanup, the remaining generators are still cleaned up and all cleanup errors are raised
        together as an :class:`ExceptionGroup`.

        Returns:
            None
        """

        exceptions: list[BaseException] = []
        for generator in reversed(self._generators):
            try:
                await self._wrap_next(generator)()
            except BaseException as cleanup_exc:
                if not isinstance(cleanup_exc, Exception):
                    # cancellation / system level exceptions must propagate immediately
                    raise
                exceptions.append(cleanup_exc)

        if exceptions:
            raise ExceptionGroup(
                "Exceptions occurred during cleanup of dependencies",
                exceptions,
            )

    async def __aenter__(self) -> None:
        """Support the async contextmanager protocol to allow for easier catching and throwing of exceptions into the
        generators.
        """

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """If an exception was raised within the contextmanager block, throw it into all generators."""
        await self.close(exc_val)

    async def _throw(self, exc: BaseException) -> None:
        """Throw an exception in all generators sequentially in reverse order of their creation.

        This mirrors the teardown order of nested context managers: the most recently established resource is
        torn down first.

        Args:
            exc: Exception to throw
        """
        exceptions: list[BaseException] = []
        for gen in reversed(self._generators):
            try:
                if isasyncgen(gen):
                    await gen.athrow(exc)
                else:
                    gen.throw(exc)  # type: ignore[union-attr]
            except (StopIteration, StopAsyncIteration):
                continue
            except BaseException as cleanup_exc:
                # a generator re-raising the exception thrown into it has cleaned itself up; continue with the
                # remaining generators so every established resource is torn down, even under cancellation
                if cleanup_exc is exc:
                    continue
                if not isinstance(cleanup_exc, Exception):
                    # non-recoverable BaseException (KeyboardInterrupt, SystemExit, a new CancelledError, ...):
                    # it must not be swallowed or wrapped in an ExceptionGroup
                    raise
                exceptions.append(cleanup_exc)

        if exceptions:
            raise ExceptionGroup(
                "Exceptions occurred during cleanup of dependencies",
                exceptions,
            ) from exc
