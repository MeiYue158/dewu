import json
from typing import Any

# SSE frame = lines of "field: value", blank line ends the frame.
# data is always json. that matters beacuse json escapes newlines, so a reply
# containing \n cannot accidentally cut the frame in half.


def frame(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def comment(text: str = "") -> str:
    """comment frame. clients ignore it by spec.

    useful to flush whatever buffer a reverse proxy is holding, and as a
    keepalive.
    """
    return f": {text}\n\n"
