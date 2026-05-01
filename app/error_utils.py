"""Helpers for keeping operational error payloads bounded."""

DEFAULT_ERROR_MESSAGE_LIMIT = 4000


def summarize_error_message(error, max_chars=DEFAULT_ERROR_MESSAGE_LIMIT):
    """Return a bounded string representation of an exception or error value."""
    if error is None:
        return None

    message = str(error)
    if len(message) <= max_chars:
        return message

    omitted = len(message) - max_chars
    return f"{message[:max_chars]}... [truncated {omitted} chars]"
