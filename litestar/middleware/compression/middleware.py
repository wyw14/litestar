# pyright: reportUnnecessaryTypeIgnoreComment=false

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

from litestar.datastructures import Headers, MutableScopeHeaders
from litestar.enums import CompressionEncoding, ScopeType
from litestar.middleware.base import AbstractMiddleware
from litestar.middleware.compression.gzip_facade import GzipCompression
from litestar.utils.empty import value_or_default
from litestar.utils.scope.state import ScopeState

if TYPE_CHECKING:
    from litestar.config.compression import CompressionConfig
    from litestar.types import (
        ASGIApp,
        HTTPResponseStartEvent,
        Message,
        Receive,
        Scope,
        Send,
    )

    try:
        from brotli import Compressor
    except ImportError:
        Compressor = Any


class CompressionMiddleware(AbstractMiddleware):
    """Compression Middleware Wrapper.

    This is a wrapper allowing for generic compression configuration / handler middleware
    """

    def __init__(self, app: ASGIApp, config: CompressionConfig) -> None:
        """Initialize ``CompressionMiddleware``

        Args:
            app: The ``next`` ASGI app to call.
            config: An instance of CompressionConfig.
        """
        super().__init__(
            app=app, exclude=config.exclude, exclude_opt_key=config.exclude_opt_key, scopes={ScopeType.HTTP}
        )
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI callable.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive function.
            send: The ASGI send function.

        Returns:
            None
        """
        accept_encoding = Headers.from_scope(scope).get("accept-encoding", "")
        config = self.config

        if config.compression_facade.encoding in accept_encoding:
            await self.app(
                scope,
                receive,
                self.create_compression_send_wrapper(
                    send=send, compression_encoding=config.compression_facade.encoding, scope=scope
                ),
            )
            return

        if config.gzip_fallback and CompressionEncoding.GZIP in accept_encoding:
            await self.app(
                scope,
                receive,
                self.create_compression_send_wrapper(
                    send=send, compression_encoding=CompressionEncoding.GZIP, scope=scope
                ),
            )
            return

        await self.app(scope, receive, send)

    def create_compression_send_wrapper(
        self,
        send: Send,
        compression_encoding: Literal[CompressionEncoding.BROTLI, CompressionEncoding.GZIP, CompressionEncoding.ZSTD]
        | str,
        scope: Scope,
    ) -> Send:
        """Wrap ``send`` to handle brotli compression.

        Args:
            send: The ASGI send function.
            compression_encoding: The compression encoding used.
            scope: The ASGI connection scope

        Returns:
            An ASGI send function.
        """
        bytes_buffer = BytesIO()

        # We can't use `self.config.compression_facade` directly if the compression is `gzip` since
        # it may be being used as a fallback.
        if compression_encoding == CompressionEncoding.GZIP:
            facade = GzipCompression(buffer=bytes_buffer, compression_encoding=compression_encoding, config=self.config)
        else:
            facade = self.config.compression_facade(  # type: ignore[assignment]
                buffer=bytes_buffer, compression_encoding=compression_encoding, config=self.config
            )

        initial_message: HTTPResponseStartEvent | None = None
        started = False
        # ``compression_active`` tracks whether the response is actually being compressed.
        # It stays ``False`` if compression was skipped (e.g. for a small response),
        # ensuring no stale compressor state or content-length is reused afterwards.
        compression_active = False

        connection_state = ScopeState.from_scope(scope)

        def close_facade() -> None:
            # close the facade, never letting a failure during cleanup mask the
            # original exception (or propagate after a successful response).
            try:
                facade.close()
            except Exception:  # noqa: BLE001
                pass

        async def send_wrapper(message: Message) -> None:
            """Handle and compresses the HTTP Message with brotli.

            Args:
                message (Message): An ASGI Message.
            """
            nonlocal started
            nonlocal initial_message
            nonlocal compression_active

            if message["type"] == "http.response.start":
                initial_message = message
                return

            if initial_message is not None and value_or_default(connection_state.is_cached, False):
                # Cached responses are already fully encoded and are passed through
                # without compression. The initial message must only ever be sent once,
                # regardless of how many body messages the cached response consists of.
                close_facade()
                await send(initial_message)
                initial_message = None
                await send(message)
                return

            if initial_message is None and value_or_default(connection_state.is_cached, False):
                # subsequent messages of a cached, multi-message response
                await send(message)
                return

            if initial_message and message["type"] == "http.disconnect":
                close_facade()
                return

            if initial_message and message["type"] == "http.response.body":
                body = message["body"]
                more_body = message.get("more_body")

                if not started:
                    started = True
                    if more_body:
                        headers = MutableScopeHeaders(initial_message)
                        headers["Content-Encoding"] = compression_encoding
                        headers.extend_header_value("vary", "Accept-Encoding")
                        del headers["Content-Length"]

                        try:
                            facade.write(body, final=not more_body)
                            compressed_body = bytes_buffer.getvalue()
                            bytes_buffer.seek(0)
                            bytes_buffer.truncate()
                        except BaseException:
                            # the response has not been started, so the error can
                            # still be reported as a regular error response.
                            close_facade()
                            raise

                        compression_active = True
                        connection_state.response_compressed = True

                        message["body"] = compressed_body
                        await send(initial_message)
                        await send(message)

                    elif len(body) >= self.config.minimum_size:
                        try:
                            facade.write(body, final=not more_body)
                            facade.close()
                            compressed_body = bytes_buffer.getvalue()
                        except BaseException:
                            # compression failed before the response started -
                            # report the error instead of sending a corrupt body.
                            close_facade()
                            raise

                        headers = MutableScopeHeaders(initial_message)
                        headers["Content-Encoding"] = compression_encoding
                        headers["Content-Length"] = str(len(compressed_body))
                        headers.extend_header_value("vary", "Accept-Encoding")
                        message["body"] = compressed_body
                        compression_active = True
                        connection_state.response_compressed = True

                        await send(initial_message)
                        await send(message)

                    else:
                        # body is too small to compress - keep the original body and
                        # content-length untouched.
                        close_facade()
                        await send(initial_message)
                        await send(message)

                elif not compression_active:
                    # compression was declined for this response, so any further
                    # messages must be passed through uncompressed to stay
                    # consistent with the response start already sent.
                    await send(message)

                else:
                    try:
                        facade.write(body, final=not more_body)
                        if not more_body:
                            facade.close()

                        compressed_body = bytes_buffer.getvalue()

                        bytes_buffer.seek(0)
                        bytes_buffer.truncate()
                    except BaseException:
                        # report the compressor failure explicitly, before the
                        # response is considered finished.
                        close_facade()
                        bytes_buffer.close()
                        raise

                    message["body"] = compressed_body

                    if not more_body:
                        bytes_buffer.close()

                    await send(message)

        return send_wrapper
