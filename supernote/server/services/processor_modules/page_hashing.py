import asyncio
import logging
from collections.abc import Sequence
from dataclasses import fields
from functools import partial
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from supernote.models.base import TaskType
from supernote.server.constants import CACHE_BUCKET, USER_DATA_BUCKET
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.models.note_processing import (
    NoteMetadataDO,
    NotePageContentDO,
    NotePageMetadataDO,
    SystemTaskDO,
)
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.services.file import FileService
from supernote.server.services.processor_modules import ProcessorModule
from supernote.server.utils.note_metadata import (
    CachedPageInk,
    ExtractedNoteMetadata,
    ExtractedPageMetadata,
    extract_note_metadata,
)
from supernote.server.utils.paths import get_page_png_path

logger = logging.getLogger(__name__)


_NOTE_EXCLUDED_FIELDS = frozenset({"pages"})


def _dataclass_column_dict(
    snapshot: ExtractedNoteMetadata | ExtractedPageMetadata,
    *,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Extract ORM column keyword arguments from an extracted metadata dataclass."""
    return {
        f.name: getattr(snapshot, f.name)
        for f in fields(snapshot)
        if f.name not in exclude
    }


def _sync_orm_fields(
    row: NoteMetadataDO | NotePageMetadataDO,
    snapshot: ExtractedNoteMetadata | ExtractedPageMetadata,
    *,
    exclude: frozenset[str] = frozenset(),
) -> None:
    """Mutate ORM attributes only when values differ to avoid redundant SQL UPDATEs."""
    for f in fields(snapshot):
        if f.name in exclude:
            continue
        new_value = getattr(snapshot, f.name)
        if getattr(row, f.name) != new_value:
            setattr(row, f.name, new_value)


class PageHashingModule(ProcessorModule):
    """Global pre-processor that extracts `.note` metadata and reconciles page hashes.

    Responsibilities:
    - Extracts notebook and per-page metadata via `extract_note_metadata()`,
      decoding bitmaps only for pages whose wire `content_hash` is not cached.
    - Persists `NoteMetadataDO` and changed `NotePageMetadataDO` rows (`0..total_pages - 1`).
    - Reconciles `NotePageContentDO` rows by `<PAGEID>` and invalidates downstream
      `SystemTaskDO` checkpoints for modified or deleted pages.
    """

    def __init__(self, file_service: FileService) -> None:
        self.file_service = file_service

    @property
    def name(self) -> str:
        return "PageHashingModule"

    @property
    def task_type(self) -> TaskType:
        return TaskType.PAGE_HASHING

    @property
    def version(self) -> str:
        """Module schema version recorded in `SystemTaskDO.data`.

        Bumping to `v2` causes `ProcessorService.recover_missing_tasks()` to
        re-run `PageHashingModule` on previously hashed `.note` files so
        `f_note_metadata` and `f_note_page_metadata` are backfilled automatically.
        """
        return "v2"

    async def run_if_needed(
        self,
        file_id: int,
        session_manager: DatabaseSessionManager,
        page_index: int | None = None,
        page_id: str | None = None,
    ) -> bool:
        """Hashing acts as the change detector, so it MUST run every time the file is processed."""
        return True

    async def cleanup_file(
        self,
        file_id: int,
        user_id: int,
        session_manager: DatabaseSessionManager,
    ) -> None:
        """Delete persisted notebook and page metadata rows for a deleted note."""
        async with session_manager.session() as session:
            await session.execute(
                delete(NoteMetadataDO).where(NoteMetadataDO.file_id == file_id)
            )
            await session.execute(
                delete(NotePageMetadataDO).where(NotePageMetadataDO.file_id == file_id)
            )
            await session.commit()

    async def process(
        self,
        file_id: int,
        session_manager: DatabaseSessionManager,
        page_index: int | None = None,
        page_id: str | None = None,
        **kwargs: object,
    ) -> None:
        """Extract incremental `.note` metadata and reconcile DB page checkpoints."""
        logger.info(f"Starting PageHashingModule for file_id={file_id}")

        async with session_manager.session() as session:
            result = await session.execute(
                select(UserFileDO).where(UserFileDO.id == file_id)
            )
            user_file = result.scalars().first()
            if not user_file:
                logger.error(f"File {file_id} not found in DB")
                return

            user_id = user_file.user_id
            storage_key = user_file.storage_key
            if not storage_key:
                logger.error(f"File {file_id} has no storage_key")
                return

            cached_rows = (
                await session.execute(
                    select(
                        NotePageMetadataDO.page_index,
                        NotePageMetadataDO.page_id,
                        NotePageMetadataDO.content_hash,
                        NotePageMetadataDO.has_ink,
                    ).where(
                        NotePageMetadataDO.user_id == user_id,
                        NotePageMetadataDO.file_id == file_id,
                    )
                )
            ).all()
            cached_pages = [
                CachedPageInk(
                    page_index=row.page_index,
                    page_id=row.page_id,
                    content_hash=row.content_hash,
                    has_ink=row.has_ink,
                )
                for row in cached_rows
                if row.content_hash
            ]

        abs_path = self.file_service.blob_storage.get_blob_path(
            USER_DATA_BUCKET, storage_key
        )
        if not abs_path.exists():
            logger.error(f"File {abs_path} does not exist on disk")
            raise FileNotFoundError(f"File {abs_path} does not exist on disk")

        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(
            None, partial(extract_note_metadata, abs_path, cached_pages)
        )

        async with session_manager.session() as session:
            await self._sync_note_and_page_metadata(
                session,
                user_id=user_id,
                file_id=file_id,
                extracted=extracted,
            )
            if extracted.pages:
                await self._sync_page_content_hashes(
                    session, file_id=file_id, pages=extracted.pages
                )
            await session.commit()

    async def _sync_note_and_page_metadata(
        self,
        session: AsyncSession,
        *,
        user_id: int,
        file_id: int,
        extracted: ExtractedNoteMetadata,
    ) -> None:
        """Upsert `NoteMetadataDO` and dirty `NotePageMetadataDO` rows, pruning truncated tail pages."""
        meta_do = await session.get(NoteMetadataDO, (user_id, file_id))
        if meta_do is None:
            session.add(
                NoteMetadataDO(
                    user_id=user_id,
                    file_id=file_id,
                    **_dataclass_column_dict(extracted, exclude=_NOTE_EXCLUDED_FIELDS),
                )
            )
        else:
            _sync_orm_fields(meta_do, extracted, exclude=_NOTE_EXCLUDED_FIELDS)

        existing_page_meta_res = await session.execute(
            select(NotePageMetadataDO).where(
                NotePageMetadataDO.user_id == user_id,
                NotePageMetadataDO.file_id == file_id,
            )
        )
        existing_page_meta = {
            pm.page_index: pm for pm in existing_page_meta_res.scalars().all()
        }

        for page in extracted.pages:
            pm = existing_page_meta.pop(page.page_index, None)
            if pm is None:
                session.add(
                    NotePageMetadataDO(
                        user_id=user_id,
                        file_id=file_id,
                        **_dataclass_column_dict(page),
                    )
                )
            else:
                _sync_orm_fields(pm, page)

        for stale_pm in existing_page_meta.values():
            await session.delete(stale_pm)

    async def _sync_page_content_hashes(
        self,
        session: AsyncSession,
        *,
        file_id: int,
        pages: Sequence[ExtractedPageMetadata],
    ) -> None:
        """Reconcile `NotePageContentDO` rows by `<PAGEID>` and invalidate stale checkpoints."""
        existing_pages = await session.execute(
            select(NotePageContentDO).where(NotePageContentDO.file_id == file_id)
        )
        existing_map = {p.page_id: p for p in existing_pages.scalars().all()}

        current_page_ids: set[str] = set()
        any_page_changed = False

        for page in pages:
            if not page.page_id:
                logger.warning(
                    f"Page {page.page_index} of file {file_id} has missing or invalid PAGEID."
                )
                continue

            current_page_ids.add(page.page_id)

            if page.page_id in existing_map:
                row = existing_map[page.page_id]

                if row.page_index != page.page_index:
                    logger.info(
                        f"Page {page.page_id} moved from {row.page_index} to {page.page_index}"
                    )
                    row.page_index = page.page_index
                    any_page_changed = True

                if row.content_hash != page.content_hash:
                    logger.info(
                        f"Page {page.page_id} (idx {page.page_index}) changed. Resetting content."
                    )
                    row.content_hash = page.content_hash
                    row.text_content = None
                    row.embedding = None
                    row.embedding_model = None
                    any_page_changed = True

                    # SystemTaskDO rows act as completion checkpoints; deleting
                    # the page's task rows causes downstream page modules (PNG,
                    # OCR, Embedding) to see no COMPLETED record in run_if_needed()
                    # and re-process this page.
                    page_task_key = f"page_{page.page_id}"
                    await session.execute(
                        delete(SystemTaskDO)
                        .where(SystemTaskDO.file_id == file_id)
                        .where(SystemTaskDO.key == page_task_key)
                    )
            else:
                logger.info(
                    f"New page {page.page_id} at index {page.page_index} detected."
                )
                any_page_changed = True
                session.add(
                    NotePageContentDO(
                        file_id=file_id,
                        page_index=page.page_index,
                        page_id=page.page_id,
                        content_hash=page.content_hash,
                        text_content=None,
                        embedding=None,
                        embedding_model=None,
                    )
                )

        for pid, row in existing_map.items():
            if pid not in current_page_ids:
                logger.info(f"Page {pid} deleted (was at {row.page_index}).")
                any_page_changed = True
                await session.delete(row)

                png_path = get_page_png_path(file_id, pid)
                try:
                    await self.file_service.blob_storage.delete(CACHE_BUCKET, png_path)
                except (FileNotFoundError, OSError) as e:
                    logger.warning(f"Failed to delete orphan PNG {pid}: {e}")

                await session.execute(
                    delete(SystemTaskDO)
                    .where(SystemTaskDO.file_id == file_id)
                    .where(SystemTaskDO.key == f"page_{pid}")
                )

        if any_page_changed:
            # Delete COMPLETED global post-processing checkpoints so file-level
            # modules re-run during the global post-processing stage (and during
            # missing-task recovery if interrupted) without PageHashingModule
            # knowing which downstream modules exist.
            await session.execute(
                delete(SystemTaskDO)
                .where(SystemTaskDO.file_id == file_id)
                .where(SystemTaskDO.key == "global")
                .where(SystemTaskDO.task_type != TaskType.PAGE_HASHING.value)
            )
