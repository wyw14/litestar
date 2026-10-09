from typing import Any

from litestar.exceptions.base_exceptions import LitestarException
from litestar.status_codes import WS_1000_NORMAL_CLOSURE, WS_1013_TRY_AGAIN_LATER

__all__ = ("WebSocketDisconnect", "WebSocketException", "WebSocketHandshakeTimeout")


class WebSocketException(LitestarException):
    """Exception class for websocket related events."""

    code: int
    """Exception code. For custom exceptions, this should be a number in the 4000+ range. Other codes can be found in
    ``litestar.status_code`` with the ``WS_`` prefix.
    """

    def __init__(self, *args: Any, detail: str, code: int = 4500) -> None:
        """Initialize ``WebSocketException``.

        Args:
            *args: Any exception args.
            detail: Exception details.
            code: Exception code. Should be a number in the >= 1000.
        """
        super().__init__(*args, detail=detail)
        self.code = code


class WebSocketDisconnect(WebSocketException):
    """Exception class for websocket disconnect events."""

    def __init__(self, *args: Any, detail: str, code: int = WS_1000_NORMAL_CLOSURE) -> None:
        """Initialize ``WebSocketDisconnect``.

        Args:
            *args: Any exception args.
            detail: Exception details.
            code: Exception code. Should be a number in the >= 1000.
        """
        super().__init__(*args, detail=detail, code=code)


class WebSocketHandshakeTimeout(WebSocketException):
    """Exception class for websocket handshake timeouts.

    Raised when the handshake (waiting for the incoming ``websocket.connect`` event) does not complete within the
    configured timeout, i.e. the upgrade was neither accepted nor rejected by the client in time.
    """

    def __init__(
        self,
        *args: Any,
        detail: str = "WebSocket handshake timed out",
        code: int = WS_1013_TRY_AGAIN_LATER,
    ) -> None:
        """Initialize ``WebSocketHandshakeTimeout``.

        Args:
            *args: Any exception args.
            detail: Exception details.
            code: Exception code. Defaults to ``1013`` (try again later) or an application provided code in the
                4000+ range.
        """
        super().__init__(*args, detail=detail, code=code)
