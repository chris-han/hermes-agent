"""Shared bounded runtime error codes for the current kernel path."""

from typing import Literal, Mapping

RuntimeContextErrorOrigin = Literal[
    "ARTIFACT_LOOKUP",
    "AUTHORITY_SESSION",
    "RETAINED_CLOSURE",
]


class RuntimeContextError(ValueError):
    """Typed runtime failure safe for service and HTTP translation."""

    def __init__(
        self,
        code: str,
        details: Mapping[str, object] | None = None,
        *,
        _origin: RuntimeContextErrorOrigin | None = None,
    ) -> None:
        self.code = code
        self.details = dict(details or {})
        self._origin = _origin
        super().__init__(code)


__all__ = ["RuntimeContextError", "RuntimeContextErrorOrigin"]
