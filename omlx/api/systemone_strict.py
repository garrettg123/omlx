# SPDX-License-Identifier: Apache-2.0
"""Strict raw-body check for ``POST /v1/systemone`` on Elephant.

Pydantic's default ``extra="ignore"`` silently drops unknown fields, and
``json.loads`` keeps the last value of a duplicate key. Elephant is text-only
and strict, so the server runs this check on the raw body before the engine
sees ``request.model_dump()``.

Stdlib only: this module must import without pydantic, fastapi, numpy or mlx.
Error messages never echo client data: they name only fixed field names.
"""

import json
from typing import Any

# Keep in sync with the field sets in omlx/api/systemone_models.py
# (SystemOneRequest and SystemOneQuestion).
REQUEST_FIELDS = frozenset({"model", "state", "questions", "images", "videos", "truncate"})
QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})

# Media fields the request model has but Elephant (text only) cannot serve.
_MEDIA_FIELDS = ("images", "videos")
# Keys a client may send expecting media or tool support. Unknown keys are
# rejected anyway; these get a clearer message.
_UNSUPPORTED_KEYS = frozenset({
    "audio", "audio_url", "input_audio", "output_audio",
    "image", "image_url", "input_image", "video", "video_url", "input_video",
    "tools", "tool_choice", "functions", "function_call",
})

class SystemOneStrictError(ValueError):
    """The raw System One body is not acceptable to Elephant."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SystemOneStrictError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(token: str) -> Any:
    raise SystemOneStrictError(f"Non-finite JSON number {token} is not allowed")


def _check_keys(obj: dict[str, Any], allowed: frozenset, where: str) -> None:
    for key in obj:
        if key in _UNSUPPORTED_KEYS:
            raise SystemOneStrictError(f"Elephant does not support {key!r} {where}")
        if key not in allowed:
            raise SystemOneStrictError(f"Unknown field {where}")


def parse_strict_systemone_body(raw: bytes | str) -> dict[str, Any]:
    """Parse a System One body strictly and return the decoded object.

    Raises ``SystemOneStrictError`` for invalid JSON, duplicate keys at any
    depth, NaN/Infinity, a non-object top level, unknown or unsupported
    fields, non-empty ``images``/``videos`` and unknown question fields.
    Type checks stay with the pydantic models.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            raise SystemOneStrictError("Request body is not valid UTF-8") from None
    if not isinstance(raw, str):
        raise SystemOneStrictError("Request body must be bytes or text")
    try:
        body = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError:
        raise SystemOneStrictError("Request body is not valid JSON") from None
    except RecursionError:
        raise SystemOneStrictError("Request body is nested too deeply") from None
    if not isinstance(body, dict):
        raise SystemOneStrictError("Request body must be a JSON object")
    _check_keys(body, REQUEST_FIELDS, "in the request")
    for key in _MEDIA_FIELDS:
        if body.get(key) not in (None, []):
            raise SystemOneStrictError(f"Elephant supports text only; {key!r} must be empty")
    questions = body.get("questions")
    if isinstance(questions, dict):
        for question in questions.values():
            if isinstance(question, dict):
                _check_keys(question, QUESTION_FIELDS, "in a question")
    return body
