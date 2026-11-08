import json
from dataclasses import replace
from pathlib import Path

import pytest

import supernote.server.utils.note_metadata as note_metadata_mod
from supernote.notebook import Notebook, load
from supernote.notebook.parser import FileObj
from supernote.notebook.schema import SupernoteMetadata
from supernote.server.utils.note_metadata import extract_note_metadata

TEST_DATA_DIR = Path(__file__).parents[2] / "testdata"


def test_extract_note_metadata_populates_note_and_page_fields() -> None:
    """Verify notebook header/footer and per-page ink/style flags are extracted."""
    note_path = TEST_DATA_DIR / "nelinory" / "A5X_TestNote_With_Pdf_Template.note"

    extracted = extract_note_metadata(note_path)

    assert extracted.file_type == "note"
    assert extracted.equipment == "A5X"
    assert extracted.width == 1404
    assert extracted.height == 1872
    assert extracted.total_pages == 3
    assert json.loads(extracted.header_json)["APPLY_EQUIPMENT"] == "A5X"
    assert "PAGE1" in json.loads(extracted.footer_json)

    assert [
        (
            p.page_index,
            p.page_style,
            p.has_ink,
            p.has_drawables,
            p.has_content,
            bool(p.content_hash),
        )
        for p in extracted.pages
    ] == [
        (0, "user_pdf_Pdf_Template_1", True, True, True, True),
        (1, "user_pdf_Pdf_Template_2", False, False, False, True),
        (2, "user_pdf_Pdf_Template_3", True, True, True, True),
    ]


def test_extract_note_metadata_loads_bitmaps_only_for_uncached_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify bitmap loading runs only for pages missing a matching content_hash."""
    note_path = TEST_DATA_DIR / "nelinory" / "A5X_TestNote_With_Pdf_Template.note"
    loaded_pages: list[int | None] = []

    def tracking_load(
        stream: FileObj,
        metadata: SupernoteMetadata | None = None,
        policy: str = "strict",
        page_number: int | None = None,
    ) -> Notebook:
        loaded_pages.append(page_number)
        return load(stream, metadata=metadata, policy=policy, page_number=page_number)

    monkeypatch.setattr(note_metadata_mod, "load", tracking_load)

    initial = extract_note_metadata(note_path)
    assert loaded_pages == [0, 1, 2]

    loaded_pages.clear()
    partial_cache = [
        replace(p.to_cached_ink(), content_hash="stale_hash")
        if p.page_index == 1
        else p.to_cached_ink()
        for p in initial.pages
    ]
    resumed = extract_note_metadata(note_path, cached_pages=partial_cache)

    assert loaded_pages == [1]
    assert resumed == initial


def test_extract_note_metadata_reuses_cached_ink_across_page_reorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify moved pages with matching (page_id, content_hash) skip bitmap decoding."""
    note_path = TEST_DATA_DIR / "nelinory" / "A5X_TestNote_With_Pdf_Template.note"
    initial = extract_note_metadata(note_path)

    loaded_pages: list[int | None] = []

    def tracking_load(
        stream: FileObj,
        metadata: SupernoteMetadata | None = None,
        policy: str = "strict",
        page_number: int | None = None,
    ) -> Notebook:
        loaded_pages.append(page_number)
        return load(stream, metadata=metadata, policy=policy, page_number=page_number)

    monkeypatch.setattr(note_metadata_mod, "load", tracking_load)

    reordered_cache = [
        replace(p.to_cached_ink(), page_index=(p.page_index + 1) % initial.total_pages)
        for p in initial.pages
    ]
    rechecked = extract_note_metadata(note_path, cached_pages=reordered_cache)

    assert loaded_pages == []
    assert rechecked == initial


def test_extract_note_metadata_caches_legacy_pages_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify legacy notes lacking <PAGEID> reuse cached has_ink by (page_index, content_hash)."""
    legacy_path = TEST_DATA_DIR / "philips" / "test.note"
    initial = extract_note_metadata(legacy_path)
    assert [p.page_id for p in initial.pages] == ["", ""]

    loaded_pages: list[int | None] = []

    def tracking_load(
        stream: FileObj,
        metadata: SupernoteMetadata | None = None,
        policy: str = "strict",
        page_number: int | None = None,
    ) -> Notebook:
        loaded_pages.append(page_number)
        return load(stream, metadata=metadata, policy=policy, page_number=page_number)

    monkeypatch.setattr(note_metadata_mod, "load", tracking_load)

    legacy_cache = [p.to_cached_ink() for p in initial.pages]
    rechecked = extract_note_metadata(legacy_path, cached_pages=legacy_cache)

    assert loaded_pages == []
    assert rechecked == initial
