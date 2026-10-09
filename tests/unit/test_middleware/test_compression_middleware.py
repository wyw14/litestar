# pyright: reportUnnecessaryTypeIgnoreComment=false

import gzip
import sys
import zlib
from collections.abc import AsyncIterator, Callable
from io import BytesIO
from typing import Literal, Union
from unittest.mock import MagicMock

import pytest

from litestar import MediaType, WebSocket, get, websocket
from litestar.config.compression import CompressionConfig
from litestar.datastructures import Headers
from litestar.enums import CompressionEncoding
from litestar.exceptions import ImproperlyConfiguredException
from litestar.handlers import HTTPRouteHandler
from litestar.middleware.compression import CompressionMiddleware
from litestar.middleware.compression.facade import CompressionFacade
from litestar.response.streaming import Stream
from litestar.status_codes import HTTP_200_OK
from litestar.testing import create_test_client
from litestar.types.asgi_types import (
    ASGIApp,
    HTTPResponseBodyEvent,
    HTTPResponseStartEvent,
    Message,
    Receive,
    Scope,
    Send,
)

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd
zstd_compression_level_upper_bound = zstd.CompressionParameter.compression_level.bounds()[1]

BrotliMode = Literal["text", "generic", "font"]


@pytest.fixture()
def handler() -> HTTPRouteHandler:
    @get(path="/", media_type=MediaType.TEXT)
    def handler_fn() -> str:
        return "_litestar_" * 4000

    return handler_fn


async def streaming_iter(content: bytes, count: int) -> AsyncIterator[bytes]:
    for _ in range(count):
        yield content


def test_compression_disabled_for_unsupported_client(handler: HTTPRouteHandler) -> None:
    with create_test_client(route_handlers=[handler], compression_config=CompressionConfig(backend="brotli")) as client:
        response = client.get("/", headers={"accept-encoding": "deflate"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 40000


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_regular_compressed_response(
    backend: Literal["gzip", "brotli", "zstd"], compression_encoding: CompressionEncoding, handler: HTTPRouteHandler
) -> None:
    with create_test_client(
        route_handlers=[handler],
        compression_config=CompressionConfig(backend=backend),
        raise_server_exceptions=True,
    ) as client:
        response = client.get("/", headers={"Accept-Encoding": str(compression_encoding.value)})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == compression_encoding
        assert int(response.headers["Content-Length"]) < 40000


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_compression_works_for_streaming_response(
    backend: Literal["gzip", "brotli", "zstd"], compression_encoding: CompressionEncoding
) -> None:
    @get("/streaming-response")
    def streaming_handler() -> Stream:
        return Stream(streaming_iter(content=b"_litestar_" * 400, count=10))

    with create_test_client(
        route_handlers=[streaming_handler], compression_config=CompressionConfig(backend=backend)
    ) as client:
        response = client.get("/streaming-response", headers={"Accept-Encoding": str(compression_encoding.value)})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == compression_encoding
        assert "Content-Length" not in response.headers


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_compression_skips_small_responses(
    backend: Literal["gzip", "brotli", "zstd"], compression_encoding: CompressionEncoding
) -> None:
    @get(path="/no-compression", media_type=MediaType.TEXT)
    def no_compress_handler() -> str:
        return "_litestar_"

    with create_test_client(
        route_handlers=[no_compress_handler], compression_config=CompressionConfig(backend=backend)
    ) as client:
        response = client.get("/no-compression", headers={"Accept-Encoding": str(compression_encoding.value)})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_"
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 10


def test_brotli_with_gzip_fallback_enabled(handler: HTTPRouteHandler) -> None:
    with create_test_client(
        route_handlers=[handler], compression_config=CompressionConfig(backend="brotli", brotli_gzip_fallback=True)
    ) as client:
        response = client.get("/", headers={"accept-encoding": CompressionEncoding.GZIP.value})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == CompressionEncoding.GZIP
        assert int(response.headers["Content-Length"]) < 40000


def test_brotli_gzip_fallback_disabled(handler: HTTPRouteHandler) -> None:
    with create_test_client(
        route_handlers=[handler],
        compression_config=CompressionConfig(backend="brotli", brotli_gzip_fallback=False),
    ) as client:
        response = client.get("/", headers={"accept-encoding": "gzip"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 40000


async def test_skips_for_websocket() -> None:
    @websocket("/")
    async def websocket_handler(socket: WebSocket) -> None:
        data = await socket.receive_json()
        await socket.send_json(data)
        await socket.close()

    with (
        create_test_client(
            route_handlers=[websocket_handler],
            compression_config=CompressionConfig(backend="brotli", brotli_gzip_fallback=False),
        ) as client,
        client.websocket_connect("/") as ws,
    ):
        assert b"content-encoding" not in dict(ws.scope["headers"])


@pytest.mark.parametrize("minimum_size, should_raise", ((0, True), (1, False), (-1, True), (100, False)))
def test_config_minimum_size_validation(minimum_size: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="brotli", brotli_gzip_fallback=False, minimum_size=minimum_size)
    else:
        CompressionConfig(backend="brotli", brotli_gzip_fallback=False, minimum_size=minimum_size)


@pytest.mark.parametrize(
    "gzip_compress_level, should_raise", ((0, False), (1, False), (-1, True), (10, True), (9, False))
)
def test_config_gzip_compress_level_validation(gzip_compress_level: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="gzip", brotli_gzip_fallback=False, gzip_compress_level=gzip_compress_level)
    else:
        CompressionConfig(backend="gzip", brotli_gzip_fallback=False, gzip_compress_level=gzip_compress_level)


@pytest.mark.parametrize(
    "zstd_compress_level, should_raise",
    (
        (-1, True),
        (0, False),
        (1, False),
        (zstd_compression_level_upper_bound, False),
        (zstd_compression_level_upper_bound + 1, True),
    ),
)
def test_config_zstd_compress_level_validation(zstd_compress_level: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="zstd", zstd_compress_level=zstd_compress_level)
    else:
        CompressionConfig(backend="zstd", zstd_compress_level=zstd_compress_level)


@pytest.mark.parametrize("brotli_quality, should_raise", ((0, False), (1, False), (-1, True), (12, True), (11, False)))
def test_config_brotli_quality_validation(brotli_quality: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_quality=brotli_quality)
    else:
        CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_quality=brotli_quality)


@pytest.mark.parametrize("brotli_lgwin, should_raise", ((9, True), (10, False), (-1, True), (25, True), (24, False)))
def test_config_brotli_lgwin_validation(brotli_lgwin: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_lgwin=brotli_lgwin)
    else:
        CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_lgwin=brotli_lgwin)


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (
        ("brotli", CompressionEncoding.BROTLI),
        ("gzip", CompressionEncoding.GZIP),
        ("zstd", CompressionEncoding.ZSTD),
    ),
)
async def test_compression_streaming_response_emitted_messages(
    backend: Literal["gzip", "brotli", "zstd"],
    compression_encoding: CompressionEncoding,
    create_scope: Callable[..., Scope],
    mock_asgi_app: ASGIApp,
) -> None:
    mock = MagicMock()

    async def fake_send(message: Message) -> None:
        mock(message)

    wrapped_send = CompressionMiddleware(
        mock_asgi_app, CompressionConfig(backend=backend)
    ).create_compression_send_wrapper(fake_send, compression_encoding, create_scope())

    await wrapped_send(HTTPResponseStartEvent(type="http.response.start", status=200, headers={}))
    # first body message always has compression headers (at least for gzip)
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"abc", more_body=True))
    # second body message with more_body=True will be empty if zlib buffers output and is not flushed
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"abc", more_body=True))
    assert mock.mock_calls[-1].args[0]["body"]
    # send a more_body=False so resources close properly
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"", more_body=False))


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_dont_recompress_cached(backend: Literal["gzip", "brotli"], compression_encoding: CompressionEncoding) -> None:
    mock = MagicMock(return_value="_litestar_" * 4000)

    @get(path="/", media_type=MediaType.TEXT, cache=True)
    def handler_fn() -> str:
        return mock()  # type: ignore[no-any-return]

    with create_test_client(
        route_handlers=[handler_fn], compression_config=CompressionConfig(backend=backend)
    ) as client:
        client.get("/", headers={"Accept-Encoding": str(compression_encoding.value)})
        response = client.get("/", headers={"Accept-Encoding": str(compression_encoding.value)})

    assert mock.call_count == 1
    assert response.status_code == HTTP_200_OK
    assert response.text == "_litestar_" * 4000
    assert response.headers["Content-Encoding"] == compression_encoding
    assert int(response.headers["Content-Length"]) < 40000


def test_compression_with_custom_backend(handler: HTTPRouteHandler) -> None:
    class ZlibCompression(CompressionFacade):
        encoding = "deflate"

        def __init__(
            self,
            buffer: BytesIO,
            compression_encoding: Union[Literal[CompressionEncoding.GZIP], str],
            config: CompressionConfig,
        ) -> None:
            self.buffer = buffer
            self.compression_encoding = compression_encoding
            self.config = config

        def write(self, body: Union[bytes, bytearray], final: bool = False) -> None:
            self.buffer.write(zlib.compress(body, level=self.config.backend_config["level"]))

        def close(self) -> None: ...

    zlib_config = {"level": 9}
    config = CompressionConfig(backend="deflate", compression_facade=ZlibCompression, backend_config=zlib_config)
    with create_test_client([handler], compression_config=config) as client:
        response = client.get("/", headers={"Accept-Encoding": "deflate"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == "deflate"
        assert int(response.headers["Content-Length"]) < 40000


def test_compression_with_custom_middleware(handler: HTTPRouteHandler) -> None:
    mock = MagicMock()

    class CustomCompressionMiddleware(CompressionMiddleware):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            mock()
            await super().__call__(scope, receive, send)
            return

    config = CompressionConfig(backend="gzip", middleware_class=CustomCompressionMiddleware)
    with create_test_client([handler], compression_config=config) as client:
        response = client.get("/", headers={"Accept-Encoding": "gzip"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == "gzip"
        assert int(response.headers["Content-Length"]) < 40000
        mock.assert_called_once()


async def test_compressed_content_length_matches_actual_body(
    create_scope: Callable[..., Scope],
    mock_asgi_app: ASGIApp,
) -> None:
    emitted: list[Message] = []

    async def fake_send(message: Message) -> None:
        emitted.append(message)

    raw_body = b"_litestar_" * 4000
    wrapped_send = CompressionMiddleware(
        mock_asgi_app, CompressionConfig(backend="gzip")
    ).create_compression_send_wrapper(fake_send, CompressionEncoding.GZIP, create_scope())

    await wrapped_send(
        HTTPResponseStartEvent(type="http.response.start", status=200, headers=[(b"content-length", b"40000")])
    )
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=raw_body, more_body=False))

    start_message, body_message = emitted
    headers = Headers(start_message["headers"])
    # the declared length must match the bytes actually put on the wire
    assert int(headers["Content-Length"]) == len(body_message["body"])
    assert gzip.decompress(body_message["body"]) == raw_body


async def test_streaming_compression_drops_content_length_and_frames_are_decodable(
    create_scope: Callable[..., Scope],
    mock_asgi_app: ASGIApp,
) -> None:
    emitted: list[Message] = []

    async def fake_send(message: Message) -> None:
        emitted.append(message)

    wrapped_send = CompressionMiddleware(
        mock_asgi_app, CompressionConfig(backend="gzip")
    ).create_compression_send_wrapper(fake_send, CompressionEncoding.GZIP, create_scope())

    await wrapped_send(
        HTTPResponseStartEvent(type="http.response.start", status=200, headers=[(b"content-length", b"40000")])
    )
    chunks = [b"_litestar_" * 200, b"_litestar_" * 200]
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=chunks[0], more_body=True))
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=chunks[1], more_body=True))
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"", more_body=False))

    start_message = emitted[0]
    headers = Headers(start_message["headers"])
    assert "Content-Length" not in headers
    assert headers["Content-Encoding"] == CompressionEncoding.GZIP
    # joining all the emitted frames must yield the complete, decodable payload
    assert gzip.decompress(b"".join(m["body"] for m in emitted[1:])) == b"_litestar_" * 400


def test_cached_streaming_response_replays_single_start() -> None:
    async def iterator() -> AsyncIterator[bytes]:
        for _ in range(3):
            yield b"_litestar_" * 400

    @get("/", cache=True, sync_to_thread=False)
    def handler_fn() -> Stream:
        return Stream(iterator())

    with create_test_client(
        route_handlers=[handler_fn], compression_config=CompressionConfig(backend="gzip")
    ) as client:
        first = client.get("/", headers={"Accept-Encoding": "gzip"})
        second = client.get("/", headers={"Accept-Encoding": "gzip"})

    assert first.status_code == HTTP_200_OK
    assert first.text == "_litestar_" * 1200
    assert second.status_code == HTTP_200_OK
    assert second.text == "_litestar_" * 1200
    assert second.headers["Content-Encoding"] == CompressionEncoding.GZIP


class _RaisingCompression(CompressionFacade):
    encoding = "x-raise"

    def __init__(
        self, buffer: BytesIO, compression_encoding: str, config: CompressionConfig
    ) -> None:
        self.buffer = buffer

    def write(self, body: Union[bytes, bytearray], final: bool = False) -> None:
        raise RuntimeError("compressor exploded")

    def close(self) -> None: ...


async def test_compressor_exception_before_start_is_raised(
    create_scope: Callable[..., Scope],
    mock_asgi_app: ASGIApp,
) -> None:
    emitted: list[Message] = []

    async def fake_send(message: Message) -> None:
        emitted.append(message)

    config = CompressionConfig(
        backend="x-raise", compression_facade=_RaisingCompression, gzip_fallback=False
    )
    wrapped_send = CompressionMiddleware(mock_asgi_app, config).create_compression_send_wrapper(
        fake_send, "x-raise", create_scope()
    )

    await wrapped_send(HTTPResponseStartEvent(type="http.response.start", status=200, headers=[]))
    # a failure on the first (uncompressed) message must surface before the
    # response starts, and nothing must have been sent
    with pytest.raises(RuntimeError, match="compressor exploded"):
        await wrapped_send(
            HTTPResponseBodyEvent(type="http.response.body", body=b"_litestar_" * 4000, more_body=False)
        )
    assert emitted == []


def test_compressor_exception_single_shot_returns_error_response(handler: HTTPRouteHandler) -> None:
    config = CompressionConfig(
        backend="x-raise", compression_facade=_RaisingCompression, gzip_fallback=False
    )
    with create_test_client([handler], compression_config=config, raise_server_exceptions=False) as client:
        response = client.get("/", headers={"Accept-Encoding": "x-raise"})
    assert response.status_code == 500
    assert "Content-Encoding" not in response.headers


async def test_further_messages_pass_through_when_compression_skipped(
    create_scope: Callable[..., Scope],
    mock_asgi_app: ASGIApp,
) -> None:
    emitted: list[Message] = []

    async def fake_send(message: Message) -> None:
        emitted.append(message)

    wrapped_send = CompressionMiddleware(
        mock_asgi_app, CompressionConfig(backend="gzip")
    ).create_compression_send_wrapper(fake_send, CompressionEncoding.GZIP, create_scope())

    await wrapped_send(
        HTTPResponseStartEvent(type="http.response.start", status=200, headers=[(b"content-length", b"10")])
    )
    # first, uncompressed (below minimum size) message - declares the response
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"_litestar_", more_body=False))
    # any further message must not touch the compressor and stay uncompressed
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"more", more_body=False))

    assert emitted[1]["body"] == b"_litestar_"
    assert emitted[2]["body"] == b"more"
