"""Incremental `.note` metadata and foreground ink extraction utilities.

Separates pure binary `.note` parsing and single-page bitmap ink scanning from
SQLAlchemy persistence and pipeline checkpoint management in `PageHashingModule`.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

from supernote.notebook import Notebook, load, parse_metadata
from supernote.notebook.schema import LinkDirection
from supernote.server.utils.hashing import get_md5_hash
from supernote.server.utils.paths import is_valid_page_id

__all__ = [
    "CachedPageInk",
    "ExtractedNoteMetadata",
    "ExtractedPageMetadata",
    "extract_note_metadata",
]


@dataclass(frozen=True)
class CachedPageInk:
    """Previously persisted `has_ink` state keyed by page identity and wire hash."""

    page_index: int
    page_id: str
    content_hash: str
    has_ink: bool


@dataclass(frozen=True)
class ExtractedPageMetadata:
    """Extracted structural metadata, ink flags, and wire JSON for a single page."""

    page_index: int
    page_id: str
    content_hash: str
    page_style: str | None
    style_hash: str | None
    orientation: str
    recogn_status: int
    has_ink: bool
    has_drawables: bool
    has_textbox: bool
    has_vector_graph: bool
    has_content: bool
    page_json: str
    keywords_json: str
    titles_json: str
    links_json: str

    def to_cached_ink(self) -> CachedPageInk:
        """Project this page's identity, hash, and ink flag into a `CachedPageInk` entry."""
        return CachedPageInk(
            page_index=self.page_index,
            page_id=self.page_id,
            content_hash=self.content_hash,
            has_ink=self.has_ink,
        )


@dataclass(frozen=True)
class ExtractedNoteMetadata:
    """Extracted notebook-level and per-page metadata from a `.note` file."""

    native_file_id: str | None
    file_type: str
    equipment: str | None
    signature: str
    width: int
    height: int
    total_pages: int
    recogn_language: str | None
    is_realtime_recognition: bool
    header_json: str
    footer_json: str
    pages: tuple[ExtractedPageMetadata, ...]


def extract_note_metadata(
    path: str | os.PathLike[str],
    cached_pages: Sequence[CachedPageInk] = (),
) -> ExtractedNoteMetadata:
    """Parse `.note` metadata in one pass and decode bitmaps only for uncached pages.

    `parse_metadata()` reads the header, footer (keywords, titles, links), and every
    page's `PageMetadata` without touching compressed bitmap blocks. Because
    `PageMetadata.to_wire_dict()` contains all `<LAYERBITMAP>`, `<LAYERINFO>`
    visibility, and legacy `<DATA>` offsets, `has_ink` cannot change while a page's
    `content_hash` stays identical—even when pages are reordered or footer
    annotations change. `load(..., page_number=i)` is therefore invoked only for
    pages missing from `cached_pages`.
    """
    ink_by_page_id: dict[tuple[str, str], bool] = {}
    ink_by_index: dict[tuple[int, str], bool] = {}
    for entry in cached_pages:
        if not entry.content_hash:
            continue
        if entry.page_id:
            ink_by_page_id[(entry.page_id, entry.content_hash)] = entry.has_ink
        else:
            ink_by_index[(entry.page_index, entry.content_hash)] = entry.has_ink

    with open(path, "rb") as stream:
        metadata = parse_metadata(stream, policy="loose")
        notebook = Notebook(metadata)

        pages: list[ExtractedPageMetadata] = []
        for i, page_info in enumerate(metadata.pages):
            page_obj = notebook.pages[i]
            page_wire_dict = page_info.to_wire_dict()
            content_hash = get_md5_hash(str(page_wire_dict))

            raw_pid = page_info.page_id
            page_id = (
                raw_pid
                if isinstance(raw_pid, str) and is_valid_page_id(raw_pid)
                else ""
            )

            cached_ink = (
                ink_by_page_id.get((page_id, content_hash))
                if page_id
                else ink_by_index.get((i, content_hash))
            )
            if cached_ink is None:
                has_ink = (
                    load(stream, metadata=metadata, policy="loose", page_number=i)
                    .page(i)
                    .has_ink
                )
            else:
                has_ink = cached_ink

            has_textbox = page_obj.has_textbox
            has_vector_graph = page_obj.has_vector_graph
            has_drawables = has_ink or has_textbox or has_vector_graph
            has_content = (
                has_drawables
                or bool(page_obj.keywords)
                or bool(page_obj.titles)
                or any(lk.direction == LinkDirection.OUT for lk in page_obj.links)
            )

            raw_orient = page_obj.orientation
            raw_recogn = page_obj.recogn_status

            pages.append(
                ExtractedPageMetadata(
                    page_index=i,
                    page_id=page_id,
                    content_hash=content_hash,
                    page_style=page_obj.style,
                    style_hash=page_obj.style_hash or None,
                    orientation=str(raw_orient) if raw_orient is not None else "1000",
                    recogn_status=int(raw_recogn) if raw_recogn is not None else 0,
                    has_ink=has_ink,
                    has_drawables=has_drawables,
                    has_textbox=has_textbox,
                    has_vector_graph=has_vector_graph,
                    has_content=has_content,
                    page_json=json.dumps(page_wire_dict, ensure_ascii=False),
                    keywords_json=json.dumps(
                        [kw.metadata.to_wire_dict() for kw in page_obj.keywords],
                        ensure_ascii=False,
                    ),
                    titles_json=json.dumps(
                        [t.metadata.to_wire_dict() for t in page_obj.titles],
                        ensure_ascii=False,
                    ),
                    links_json=json.dumps(
                        [lk.metadata.to_wire_dict() for lk in page_obj.links],
                        ensure_ascii=False,
                    ),
                )
            )

    raw_equip = metadata.header.apply_equipment
    return ExtractedNoteMetadata(
        native_file_id=notebook.file_id,
        file_type=notebook.type,
        equipment=str(raw_equip) if raw_equip is not None else None,
        signature=notebook.signature,
        width=notebook.width,
        height=notebook.height,
        total_pages=len(pages),
        recogn_language=metadata.header.file_recogn_language,
        is_realtime_recognition=notebook.is_realtime_recognition(),
        header_json=json.dumps(metadata.header.to_wire_dict(), ensure_ascii=False),
        footer_json=json.dumps(metadata.footer.to_wire_dict(), ensure_ascii=False),
        pages=tuple(pages),
    )
