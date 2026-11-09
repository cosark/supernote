"""Pure text normalization, page timestamp inference, length formatting, and deep-link helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from supernote.models.schedule import ScheduleTaskLinkDTO
from supernote.server.rapid_log.models import (
    MAX_TASK_DETAIL_LENGTH,
    MAX_TASK_TITLE_LENGTH,
)
from supernote.server.utils.rapid_log import RapidLogTaskState

__all__ = [
    "build_note_task_link",
    "expected_importance",
    "expected_status",
    "field_values_differ",
    "format_task_title_and_detail",
    "infer_page_created_ms",
    "infer_page_date_ms",
    "merge_migrated_detail",
    "normalize_due_time",
    "normalize_rapid_log_text",
]

_WHITESPACE_RE = re.compile(r"\s+")
_PAGE_CREATED_HMS_RE = re.compile(r"^P(\d{14})")
_PAGE_CREATED_DATE_RE = re.compile(r"^P(\d{8})")


def normalize_rapid_log_text(text: str) -> str:
    """Normalize bullet text for idempotent matching across syncs and migrations."""
    return _WHITESPACE_RE.sub(" ", text.strip()).casefold()


def infer_page_date_ms(page_id: str) -> int | None:
    """Infer UTC start-of-day epoch milliseconds from a Supernote `P<YYYYMMDD>...` page ID."""
    if not page_id or not page_id.startswith("P"):
        return None
    match = _PAGE_CREATED_DATE_RE.match(page_id)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def infer_page_created_ms(page_id: str) -> int | None:
    """Infer UTC page creation epoch milliseconds from `P<YYYYMMDDHHMMSS>...` (or `P<YYYYMMDD>...`)."""
    if not page_id or not page_id.startswith("P"):
        return None
    hms_match = _PAGE_CREATED_HMS_RE.match(page_id)
    if hms_match:
        try:
            dt = datetime.strptime(hms_match.group(1), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    return infer_page_date_ms(page_id)


def format_task_title_and_detail(
    text: str,
    detail: str | None,
    *,
    max_title_length: int = MAX_TASK_TITLE_LENGTH,
    max_detail_length: int = MAX_TASK_DETAIL_LENGTH,
) -> tuple[str, str]:
    """Format `(title, detail)` safely within `ScheduleService` length limits.

    If `text` exceeds `max_title_length`, truncates `title` with `…` and prepends
    the full untruncated `text` into `detail` so zero handwritten content is lost.
    """
    cleaned_text = _WHITESPACE_RE.sub(" ", text.strip())
    raw_detail = (detail or "").strip()

    if len(cleaned_text) <= max_title_length:
        title = cleaned_text
        combined_detail = raw_detail
    else:
        title = cleaned_text[: max_title_length - 1].rstrip() + "…"
        combined_detail = (
            f"{cleaned_text}\n\n{raw_detail}" if raw_detail else cleaned_text
        )

    if len(combined_detail) > max_detail_length:
        combined_detail = combined_detail[: max_detail_length - 1] + "…"

    return title, combined_detail


def merge_migrated_detail(
    existing_detail: str | None,
    new_item_detail: str | None,
    *,
    max_detail_length: int = MAX_TASK_DETAIL_LENGTH,
) -> str:
    """Preserve prior context notes when migrating a task headline, appending new sub-notes if added."""
    existing_clean = (existing_detail or "").strip()
    new_clean = (new_item_detail or "").strip()
    if not new_clean:
        return existing_clean
    if not existing_clean or new_clean == existing_clean or new_clean in existing_clean:
        return existing_clean if new_clean in existing_clean else new_clean
    if existing_clean in new_clean:
        return new_clean
    merged = f"{existing_clean}\n{new_clean}"
    if len(merged) > max_detail_length:
        merged = merged[: max_detail_length - 1] + "…"
    return merged


def build_note_task_link(
    *,
    file_id: int,
    file_path: str,
    page_index: int,
    page_id: str,
    task_title: str,
) -> str:
    """Encode a Base64 `ScheduleTaskLinkDTO` deep-link to a 1-indexed `.note` page."""
    return ScheduleTaskLinkDTO(
        app_name="Note",
        path=file_path,
        page=page_index + 1,
        file_id=str(file_id),
        page_id=page_id,
        task_title=task_title,
    ).to_b64()


def normalize_due_time(due_time: int | None) -> int | None:
    """Normalize `0` and `None` as unset due time."""
    return due_time if due_time else None


def expected_importance(is_starred: bool) -> str:
    """Return expected task importance for a starred/unstarred bullet."""
    return "high" if is_starred else "normal"


def expected_status(task_state: RapidLogTaskState | None) -> str:
    """Return expected `ScheduleTask` status string for a `RapidLogTaskState`."""
    return "completed" if task_state == RapidLogTaskState.COMPLETED else "needsAction"


def field_values_differ(field_name: str, a: Any, b: Any) -> bool:
    """Compare two `ScheduleTask` field values accounting for nullable/zero conventions."""
    if field_name == "due_time":
        return normalize_due_time(a) != normalize_due_time(b)
    if field_name == "detail":
        return (a or "") != (b or "")
    return bool(a != b)
