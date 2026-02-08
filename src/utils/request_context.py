"""Request context for passing client IP and other request-scoped data."""

from contextvars import ContextVar

# Client IP from the incoming HTTP request (for IP-based location, e.g. weather)
client_ip_context: ContextVar[str] = ContextVar("client_ip", default="")


def get_client_ip() -> str:
    """Get the client IP from the current request context."""
    return client_ip_context.get() or ""


def set_client_ip(ip: str) -> None:
    """Set the client IP for the current request context."""
    client_ip_context.set(ip or "")
