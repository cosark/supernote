"""Top-level Rapid Logging lifecycle state machine and notebook reconciliation planner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from supernote.server.rapid_log.actions import ActionAccumulator
from supernote.server.rapid_log.formatting import (
    build_note_task_link,
    expected_importance,
    expected_status,
    format_task_title_and_detail,
    infer_page_created_ms,
    infer_page_date_ms,
    merge_migrated_detail,
    normalize_due_time,
)
from supernote.server.rapid_log.matching import (
    EXPLICIT_MIGRATION_STATES,
    OPEN_TASK_STATES,
    ParsedCandidate,
    ResolvedPage,
    find_migration_source,
    match_page_candidates,
)
from supernote.server.rapid_log.models import (
    DeleteTaskAction,
    EntryLinkState,
    NotebookPageInput,
    ReconciliationPlan,
    RemoteTaskSnapshot,
    SyncedRapidLogEntry,
)
from supernote.server.utils.rapid_log import (
    RapidLogKind,
    RapidLogTaskState,
    parse_rapid_log,
)
from supernote.server.utils.unique_id import next_id

__all__ = [
    "plan_notebook_deletion",
    "reconcile_rapid_log",
]


def _default_task_id_factory() -> str:
    return str(next_id())


def _build_synced_entry(
    cand: ParsedCandidate,
    *,
    link_state: EntryLinkState,
    schedule_task_id: str | None,
    last_synced_ms: int,
    entry_id: int | str | None = None,
) -> SyncedRapidLogEntry:
    rp = cand.page
    item = cand.item
    return SyncedRapidLogEntry(
        file_id=rp.file_id,
        file_path=rp.file_path,
        page_id=rp.page_id,
        page_index=rp.page_index,
        item_order=cand.raw_order,
        occurrence_index=cand.occurrence_index,
        kind=item.kind,
        normalized_text=cand.normalized_text,
        text=item.text,
        detail=item.detail,
        task_state=item.task_state if item.kind == RapidLogKind.TASK else None,
        is_starred=item.is_starred,
        is_cancelled=item.is_cancelled,
        page_date_ms=rp.page_date_ms,
        page_created_ms=rp.page_created_ms,
        link_state=link_state,
        schedule_task_id=schedule_task_id,
        last_synced_ms=last_synced_ms,
        entry_id=entry_id,
    )


def _reconcile_matched_entry(
    cand: ParsedCandidate,
    prev_entry: SyncedRapidLogEntry,
    acc: ActionAccumulator,
    sync_time_ms: int,
) -> SyncedRapidLogEntry:
    """Reconcile a matched item against `prev_entry` and `acc.effective_tasks` (3-way diff)."""
    rp = cand.page
    item = cand.item
    page_mod_ms = rp.page_modified_ms or sync_time_ms
    new_synced_ms = max(sync_time_ms, page_mod_ms, prev_entry.last_synced_ms)

    if item.kind != RapidLogKind.TASK:
        return _build_synced_entry(
            cand,
            link_state=EntryLinkState.UNLINKED,
            schedule_task_id=None,
            last_synced_ms=new_synced_ms,
            entry_id=prev_entry.entry_id,
        )

    curr_title, curr_detail = format_task_title_and_detail(item.text, item.detail)
    prev_title, prev_detail = format_task_title_and_detail(
        prev_entry.text, prev_entry.detail
    )
    curr_links = build_note_task_link(
        file_id=rp.file_id,
        file_path=rp.file_path,
        page_index=rp.page_index,
        page_id=rp.page_id,
        task_title=curr_title,
    )

    link_state = prev_entry.link_state
    task_id = prev_entry.schedule_task_id

    if link_state == EntryLinkState.ACTIVE and task_id is not None:
        remote = acc.effective_tasks.get(task_id)
        if remote is None:
            link_state = EntryLinkState.DISMISSED_DIGITAL
            task_id = None
        elif item.is_cancelled or item.task_state == RapidLogTaskState.CANCELLED:
            if remote.status != "completed":
                acc.record_delete(task_id, reason="cancelled_ink")
            link_state = EntryLinkState.CANCELLED_INK
            task_id = None
        else:
            updates: dict[str, Any] = {}
            reasons: list[str] = []

            # 1. Status (`needsAction` <-> `completed`)
            prev_comp = prev_entry.task_state == RapidLogTaskState.COMPLETED
            curr_comp = item.task_state == RapidLogTaskState.COMPLETED
            if prev_comp != curr_comp:
                desired_status = "completed" if curr_comp else "needsAction"
                digital_status_changed = remote.status != expected_status(
                    prev_entry.task_state
                )
                if remote.status != desired_status and (
                    not digital_status_changed or page_mod_ms >= remote.update_time
                ):
                    updates["status"] = desired_status
                    updates["completed_time"] = (
                        max(page_mod_ms, sync_time_ms) if curr_comp else 0
                    )
                    reasons.append("ink_completed" if curr_comp else "ink_reopened")

            # 2. Importance (`* ` starred <-> normal)
            if prev_entry.is_starred != item.is_starred:
                digital_imp_changed = (
                    remote.importance or "normal"
                ) != expected_importance(prev_entry.is_starred)
                if not digital_imp_changed or page_mod_ms >= remote.update_time:
                    updates["importance"] = expected_importance(item.is_starred)
                    reasons.append("ink_importance")

            # 3. Title
            if curr_title != prev_title:
                if remote.title == prev_title or page_mod_ms >= remote.update_time:
                    updates["title"] = curr_title
                    reasons.append("ink_title")

            # 4. Detail (preserve prior migrated or manual prefix notes when sub-bullets grow/edit)
            if curr_detail != prev_detail:
                remote_detail_str = remote.detail or ""
                if remote_detail_str == prev_detail:
                    updates["detail"] = curr_detail
                    reasons.append("ink_detail")
                elif prev_detail and remote_detail_str.endswith("\n" + prev_detail):
                    prefix = remote_detail_str[: -(len(prev_detail) + 1)]
                    updates["detail"] = (
                        f"{prefix}\n{curr_detail}" if curr_detail else prefix
                    )
                    reasons.append("ink_detail")
                elif not prev_detail and curr_detail and remote_detail_str:
                    updates["detail"] = merge_migrated_detail(
                        remote_detail_str, curr_detail
                    )
                    reasons.append("ink_detail")
                elif page_mod_ms >= remote.update_time:
                    updates["detail"] = curr_detail
                    reasons.append("ink_detail")

            # 5. Deep-link (`links`): owned by the notebook anchor
            desired_links = build_note_task_link(
                file_id=rp.file_id,
                file_path=rp.file_path,
                page_index=rp.page_index,
                page_id=rp.page_id,
                task_title=updates.get("title", remote.title),
            )
            if remote.links != desired_links:
                updates["links"] = desired_links
                reasons.append("anchor_updated")

            if updates:
                acc.record_update(
                    task_id,
                    updates,
                    reason="+".join(reasons),
                    modified_ms=page_mod_ms,
                )
            updated_remote = acc.effective_tasks.get(task_id)
            if updated_remote is not None:
                new_synced_ms = max(new_synced_ms, updated_remote.update_time)

    elif link_state in (EntryLinkState.UNLINKED, EntryLinkState.CANCELLED_INK):
        if not item.is_cancelled and item.task_state in OPEN_TASK_STATES:
            task_id = acc.record_create(
                title=curr_title,
                detail=curr_detail,
                importance=expected_importance(item.is_starred),
                due_time=rp.page_date_ms,
                links=curr_links,
                modified_ms=page_mod_ms,
            )
            link_state = EntryLinkState.ACTIVE
            new_synced_ms = max(new_synced_ms, acc.effective_tasks[task_id].update_time)

    elif link_state == EntryLinkState.DISMISSED_DIGITAL:
        if (
            prev_entry.task_state
            in (RapidLogTaskState.COMPLETED, RapidLogTaskState.CANCELLED)
            and item.task_state == RapidLogTaskState.INCOMPLETE
            and not item.is_cancelled
        ):
            task_id = acc.record_create(
                title=curr_title,
                detail=curr_detail,
                importance=expected_importance(item.is_starred),
                due_time=rp.page_date_ms,
                links=curr_links,
                modified_ms=page_mod_ms,
            )
            link_state = EntryLinkState.ACTIVE
            new_synced_ms = max(new_synced_ms, acc.effective_tasks[task_id].update_time)

    return _build_synced_entry(
        cand,
        link_state=link_state,
        schedule_task_id=task_id,
        last_synced_ms=new_synced_ms,
        entry_id=prev_entry.entry_id,
    )


def plan_notebook_deletion(
    entries: Sequence[SyncedRapidLogEntry],
    remote_tasks: Mapping[str, RemoteTaskSnapshot],
) -> list[DeleteTaskAction]:
    """Plan `ScheduleTask` deletions when an entire `.note` file is deleted.

    Deletes still-open (`needsAction`) linked tasks while preserving `completed` tasks
    as historical records.
    """
    deletions: list[DeleteTaskAction] = []
    seen_task_ids: set[str] = set()
    for entry in entries:
        task_id = entry.schedule_task_id
        if (
            entry.link_state != EntryLinkState.ACTIVE
            or not task_id
            or task_id in seen_task_ids
        ):
            continue
        remote = remote_tasks.get(task_id)
        if remote is not None and remote.status != "completed":
            seen_task_ids.add(task_id)
            deletions.append(
                DeleteTaskAction(task_id=task_id, reason="notebook_deleted")
            )
    return deletions


def reconcile_rapid_log(
    *,
    file_id: int,
    file_path: str,
    pages: Sequence[NotebookPageInput],
    prev_entries: Sequence[SyncedRapidLogEntry],
    remote_tasks: Mapping[str, RemoteTaskSnapshot],
    sync_time_ms: int,
    external_entries: Sequence[SyncedRapidLogEntry] = (),
    task_id_factory: Callable[[], str] = _default_task_id_factory,
) -> ReconciliationPlan:
    """Reconcile a notebook's OCR pages against previous state and digital `ScheduleTask` snapshots."""
    # Step 0: Resolve page metadata & sort chronologically (`P<timestamp>` when present, else `page_index`)
    raw_pages = [
        ResolvedPage(
            file_id=file_id,
            file_path=file_path,
            page_id=p.page_id,
            page_index=p.page_index,
            page_modified_ms=p.page_modified_ms,
            page_date_ms=(
                p.page_date_ms
                if p.page_date_ms is not None
                else infer_page_date_ms(p.page_id)
            ),
            page_created_ms=(
                p.page_created_ms
                if p.page_created_ms is not None
                else infer_page_created_ms(p.page_id)
            ),
            chrono_rank=0,
            items=(
                p.items
                if p.items is not None
                else tuple(parse_rapid_log(p.text_content))
            ),
        )
        for p in pages
    ]
    chrono_pages = sorted(
        raw_pages,
        key=lambda rp: (
            0 if rp.page_created_ms is None else 1,
            rp.page_created_ms or 0,
            rp.page_index,
            rp.page_id,
        ),
    )
    for rank, rp in enumerate(chrono_pages):
        rp.chrono_rank = rank

    # Mark entries whose digital To-Do was manually deleted as `DISMISSED_DIGITAL`
    def _normalize_baseline(entry: SyncedRapidLogEntry) -> SyncedRapidLogEntry:
        if (
            entry.link_state == EntryLinkState.ACTIVE
            and entry.schedule_task_id is not None
            and entry.schedule_task_id not in remote_tasks
        ):
            return replace(
                entry,
                link_state=EntryLinkState.DISMISSED_DIGITAL,
                schedule_task_id=None,
            )
        return entry

    prev_by_page: dict[str, list[SyncedRapidLogEntry]] = defaultdict(list)
    for entry in prev_entries:
        if entry.file_id == file_id:
            prev_by_page[entry.page_id].append(_normalize_baseline(entry))
    for page_list in prev_by_page.values():
        page_list.sort(key=lambda e: (e.item_order, e.occurrence_index))

    working_external = [
        _normalize_baseline(e) for e in external_entries if e.file_id != file_id
    ]
    external_dirty_indices: set[int] = set()

    # Step 1: Match current page items against previous entries (exact Pass 1 + fuzzy Pass 2)
    matched_pairs: list[tuple[ParsedCandidate, SyncedRapidLogEntry]] = []
    unmatched_new_candidates: list[ParsedCandidate] = []
    unmatched_prev_entries: list[SyncedRapidLogEntry] = []

    present_page_ids = {rp.page_id for rp in chrono_pages}
    for page_id, prev_list in prev_by_page.items():
        if page_id not in present_page_ids:
            unmatched_prev_entries.extend(prev_list)

    for rp in chrono_pages:
        m_pairs, u_new, u_prev = match_page_candidates(
            rp, prev_by_page.get(rp.page_id, [])
        )
        matched_pairs.extend(m_pairs)
        unmatched_new_candidates.extend(u_new)
        unmatched_prev_entries.extend(u_prev)

    acc = ActionAccumulator(remote_tasks, sync_time_ms, task_id_factory)

    # Step 2: Reconcile all matched entries on their existing pages first
    working_entries = [
        _reconcile_matched_entry(cand, prev_entry, acc, sync_time_ms)
        for cand, prev_entry in matched_pairs
    ]

    # Step 3: Reconcile brand-new items (migrations, lasso relocations, or fresh task creations)
    handed_off_prev_ids: set[int] = set()
    ordered_new = sorted(
        unmatched_new_candidates,
        key=lambda c: (
            0
            if c.item.kind == RapidLogKind.TASK
            and c.item.task_state in EXPLICIT_MIGRATION_STATES
            else 1,
            c.page.chrono_rank,
            c.raw_order,
        ),
    )

    for cand in ordered_new:
        rp = cand.page
        item = cand.item
        page_mod_ms = rp.page_modified_ms or sync_time_ms
        new_synced_ms = max(sync_time_ms, page_mod_ms)

        if item.kind != RapidLogKind.TASK:
            working_entries.append(
                _build_synced_entry(
                    cand,
                    link_state=EntryLinkState.UNLINKED,
                    schedule_task_id=None,
                    last_synced_ms=new_synced_ms,
                )
            )
            continue

        curr_title, curr_detail = format_task_title_and_detail(item.text, item.detail)
        link_state = EntryLinkState.UNLINKED
        task_id: str | None = None

        if not item.is_cancelled and item.task_state != RapidLogTaskState.CANCELLED:
            mig_match = find_migration_source(
                cand,
                working_entries=working_entries,
                unmatched_prev_entries=unmatched_prev_entries,
                handed_off_prev_ids=handed_off_prev_ids,
                working_external=working_external,
                effective_tasks=acc.effective_tasks,
            )
            if mig_match is not None:
                pool_name, pool_idx, src_entry = mig_match
                adopted_task_id = src_entry.schedule_task_id
                assert adopted_task_id is not None
                remote = acc.effective_tasks[adopted_task_id]

                retired_src = replace(
                    src_entry,
                    link_state=EntryLinkState.MIGRATED,
                    schedule_task_id=None,
                    last_synced_ms=max(src_entry.last_synced_ms, sync_time_ms),
                )
                if pool_name == "working":
                    working_entries[pool_idx] = retired_src
                elif pool_name == "unmatched_prev":
                    handed_off_prev_ids.add(pool_idx)
                elif pool_name == "external":
                    working_external[pool_idx] = retired_src
                    external_dirty_indices.add(pool_idx)

                handoff_updates: dict[str, Any] = {
                    "links": build_note_task_link(
                        file_id=file_id,
                        file_path=file_path,
                        page_index=rp.page_index,
                        page_id=rp.page_id,
                        task_title=remote.title,
                    )
                }
                merged_detail = merge_migrated_detail(remote.detail, curr_detail)
                if merged_detail != (remote.detail or ""):
                    handoff_updates["detail"] = merged_detail
                if item.is_starred and remote.importance != "high":
                    handoff_updates["importance"] = "high"

                curr_remote_due = normalize_due_time(remote.due_time)
                if rp.page_date_ms is not None and (
                    curr_remote_due == normalize_due_time(src_entry.page_date_ms)
                    or (
                        curr_remote_due is not None
                        and curr_remote_due < rp.page_date_ms
                    )
                    or (src_entry.page_date_ms is None and curr_remote_due is None)
                ):
                    handoff_updates["due_time"] = rp.page_date_ms

                if item.task_state == RapidLogTaskState.COMPLETED:
                    handoff_updates["status"] = "completed"
                    handoff_updates["completed_time"] = max(page_mod_ms, sync_time_ms)

                acc.record_update(
                    adopted_task_id,
                    handoff_updates,
                    reason="migrated_handoff",
                    modified_ms=page_mod_ms,
                )
                task_id = adopted_task_id
                link_state = EntryLinkState.ACTIVE
                new_synced_ms = max(
                    new_synced_ms, acc.effective_tasks[adopted_task_id].update_time
                )

            elif item.task_state in OPEN_TASK_STATES:
                task_id = acc.record_create(
                    title=curr_title,
                    detail=curr_detail,
                    importance=expected_importance(item.is_starred),
                    due_time=rp.page_date_ms,
                    links=build_note_task_link(
                        file_id=file_id,
                        file_path=file_path,
                        page_index=rp.page_index,
                        page_id=rp.page_id,
                        task_title=curr_title,
                    ),
                    modified_ms=page_mod_ms,
                )
                link_state = EntryLinkState.ACTIVE
                new_synced_ms = max(
                    new_synced_ms, acc.effective_tasks[task_id].update_time
                )

        working_entries.append(
            _build_synced_entry(
                cand,
                link_state=link_state,
                schedule_task_id=task_id,
                last_synced_ms=new_synced_ms,
            )
        )

    # Step 4: Clean up erased open tasks (preserving completed tasks as historical record)
    for idx, prev_entry in enumerate(unmatched_prev_entries):
        if (
            idx not in handed_off_prev_ids
            and prev_entry.link_state == EntryLinkState.ACTIVE
            and prev_entry.schedule_task_id is not None
        ):
            remote = acc.effective_tasks.get(prev_entry.schedule_task_id)
            if remote is not None and remote.status != "completed":
                acc.record_delete(
                    prev_entry.schedule_task_id, reason="erased_from_page"
                )

    working_entries.sort(key=lambda e: (e.page_index, e.page_id, e.item_order))
    updated_external = [working_external[i] for i in sorted(external_dirty_indices)]
    create_actions, update_actions, delete_actions = acc.finalize_actions(
        working_entries
    )

    return ReconciliationPlan(
        entries=working_entries,
        updated_external_entries=updated_external,
        create_tasks=create_actions,
        update_tasks=update_actions,
        delete_tasks=delete_actions,
    )
