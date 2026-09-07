# AUDIT.md — Hostile Production Audit of RAG-Assistant

**Date:** 2026-09-04
**Scope:** full repository — `server/app` (FastAPI + clean architecture), `server/app/infrastructure` (SQLAlchemy, Qdrant, S3/MinIO, Redis, arq, APScheduler), migrations, docker-compose, Dockerfile, CI, tests (648), loadtest.
**Method:** static review of all execution paths (HTTP → FastAPI → DI → services → domain → repositories → infra), cross-check of subagent deep-dives with direct code reading, read-only runtime verification (`pytest`, targeted greps). No source code was modified.
**Confidence legend:** `CONFIRMED` (verified by reading code; failure path fully traced), `HIGHLY LIKELY` (code-verified, manifestation depends on deployment), `POTENTIAL` (requires runtime/cluster-mode verification).

---

# Executive Summary

The project shows genuine clean-architecture discipline (explicit ports, UoW, transactional outbox, LISTEN/NOTIFY config propagation, structured logging) — clearly above average for this class of system. However, the audit found **critical authorization breaches in the two flagship retrieval paths** (exact search and hybrid RAG), **several event-loop blocking operations that can freeze the entire API**, **broken background-job failure semantics**, and **a permanently red test suite (10 failing tests)** that makes the CI quality gate meaningless.

Top problems:

1. **Any authenticated user (including external API-key clients) can read private chunks of any other user/tenant** via `POST /search/exact` — the SQL ACL is built with `OR` instead of `AND`, and `document_id` is inside the same `OR` group, making it a total bypass. The same broken repository is used inside `/chat` for the exact-reference boost.
2. **Hybrid (BM25) search resolves chunk hashes without the ACL filter** — private chunks of other users flow into LLM context and `sources`.
3. **Synchronous OCR / PDF analysis / S3 / Qdrant calls run directly on the asyncio event loop** — one admin dry-run with OCR freezes the whole API (single uvicorn process).
4. **arq jobs are silently killed at 300 s** (default `job_timeout`, no explicit override) and `CancelledError` escapes every `except Exception` → documents stay `PROCESSING` forever; the orphan reaper then marks still-running jobs as failed.
5. **BM25 index is per-process** — documents ingested by the worker never reach the API process's sparse index until API restart.

Scores are at the end; remediation roadmap is at the very end.

---

# Architecture Overview

Verified flow (not inferred from names):

```
client (nginx :3001, /api/* proxy)
  └─► FastAPI (uvicorn, SINGLE process, :8001 also published directly on host)
        ├─ middlewares: CORS, RequestID (BaseHTTP), MetricsMiddleware (BaseHTTP), Instrumentator (/metrics unauthenticated)
        ├─ auth: JWT (HS256, 24h) or Api-Key (sha256 → Redis cache 30 s + pub/sub revoke)
        ├─ Depends providers (presentation/api/dependencies.py) → Container (built once in lifespan)
        │     ├─ InfrastructureContainer: UoW factory, repos, ML registry, storage, listeners, outbox dispatcher
        │     └─ ApplicationContainer: ~20 services
        ├─ Application services (each method opens its own UnitOfWork)
        ├─ UnitOfWork (commit at context exit; master/slave session selection)
        ├─ Domain: entities (dataclasses), VOs, access_control (canonical ACL conditions), parser protocol
        └─ Repositories: SQLAlchemy → PostgreSQL 16; Qdrant client; S3 (boto3, sync)
Background:
  ├─ arq worker (separate process, max_jobs=4, cron: cleanup/recover/BM25 rebuild)
  ├─ APScheduler in API process: metrics (30 s), config resync (5 min), outbox dispatch (30 s), reconcile (60 s)
  └─ Postgres LISTEN/NOTIFY: vector_outbox_ready → outbox listener (API + worker); config_changed → config listener
Storage/infra: PostgreSQL (timestamptz + naive mix), Qdrant (single collection, ACL payload indexes),
MinIO (documents, previews, BM25 index), Redis (rate limits, api-key cache, answer cache, arq, SSE sweeps)
```

Positives worth keeping: transactional outbox with `FOR UPDATE SKIP LOCKED` claim + backoff + dead-letter; NOTIFY sent inside the outbox transaction; config listener with supervisor + reconnect resync; `ensure_admin` with `ON CONFLICT DO NOTHING`; bcrypt offloaded to thread pool; JWT role re-read from DB (stale-role claims don't work); API-key cache invalidated on revoke (30 s TTL).

---

# Critical Findings

## [CRITICAL] C-1. SQL ACL in `search_substring` is built with OR instead of AND — full cross-user/cross-tenant read, and `document_id` fully bypasses ACL

**Category:** Security (BOLA / broken access control)
**Confidence:** CONFIRMED (verified by reading and mentally compiling the SQLAlchemy clause; consistent with the two correct reference implementations in the same codebase)
**Location:** `server/app/infrastructure/repositories/sqlalchemy_chunk_repository.py:170-183, 204, 227`
Contract violated: `server/app/domain/services/access_control.py:29-41` — "Each condition is an **AND**-clause. The full filter is OR of all conditions."

### Problem

```python
# lines 170-180
acl_clauses = []
for cond in conditions:
    parts = [ChunkModel.visibility == cond.visibility.value]
    if cond.owner_match == OwnerMatch.SELF.value:
        parts.append(ChunkModel.owner_id == user["id"])
    if cond.group_match:
        parts.append(ChunkModel.group_id.in_(group_ids))
    acl_clauses.append(or_(*parts))          # ← must be and_(*parts)

# lines 182-183
if document_id is not None:
    acl_clauses.append(ChunkModel.document_id == document_id)

# line 204 / 227
.where(or_(*acl_clauses) if acl_clauses else text("true"))
```

Each per-condition AND-group is collapsed into an OR. For a client user the generated predicate is
`visibility = 'client_private' OR owner_id = <self>` — every other client's private chunk matches the first arm. For an internal user: `visibility = 'internal_private' OR owner_id = <self>`, and `visibility = 'internal_group' OR group_id IN (...)`. Additionally, `document_id == X` is **inside the final OR**, so the predicate `OR (document_id = X)` matches any row of any document regardless of visibility/owner.

The correct implementation exists 30 lines away (`sqlalchemy_document_repository.py:172-182` uses `and_(*and_parts)` with `document_id` handled separately) and in the Qdrant filter (`infrastructure/acl.py:47-58`, `must` inside / `should` between) — this third copy drifted.

### Why it matters

- Cross-tenant confidentiality breach: any authenticated account (including a lowest-privilege `kind=client` API key) reads private content of other clients/users/groups.
- Scripted bulk exfiltration: endpoint is unthrottled, `limit` up to 100/req, `mode="icontains"` accepts any ≥3-char substring.
- The same repository is injected into `/chat` via `ChunkSearchAdapter.search_substring` (`infrastructure/adapters/chunk_search_adapter.py:22-30`), called from `RagService._apply_exact_search` (`infrastructure/ml/rag_service.py:342`, triggered at `:693` when the exact-ref boost fires) → private chunks of other users are appended to LLM candidates and returned in `sources`.

### Reproduction / scenario

1. Attacker obtains any JWT or any client API key.
2. `POST /search/exact {"query": "а", "mode": "icontains", "document_id": 5, "limit": 100}` (and paginate by unique terms/offsets) → chunks of document 5 are returned with full `content`, even if document 5 is another user's `internal_private` or another client's `client_private`.
3. Alternatively, without `document_id`: client A queries a distinctive term from client B's contract → matched by the `visibility='client_private'` arm.
4. Via chat: craft a question containing a verbatim reference from a victim document → exact boost injects victim chunk into context; answer quotes it.

### Evidence

- `sqlalchemy_chunk_repository.py:180` `acl_clauses.append(or_(*parts))`
- `sqlalchemy_chunk_repository.py:183` `acl_clauses.append(ChunkModel.document_id == document_id)`
- `sqlalchemy_chunk_repository.py:204` `.where(or_(*acl_clauses) ...)`
- Correct counterpart: `sqlalchemy_document_repository.py:174-182` (`and_parts` → `and_(*and_parts)`)
- Canonical contract: `access_control.py:29-41`

### Fix

Build `and_(*parts)` per condition; keep `document_id` as a separate top-level `and_(...)` outside the OR. Add a compiled-SQL assertion test (compile the statement and assert `visibility = ... AND owner_id = ...`) plus a behavioral test: client A must not retrieve chunks with `visibility='client_private'` and `owner_id != A`; internal non-member must not retrieve `internal_group` chunks.

### Tests

None today. `tests/test_acl.py` tests only the domain service and Qdrant filter builder; the SQL translation is exercised exclusively with fakes (`tests/fakes.py:133` mocks `search_substring`) — which is exactly why this bug shipped.

---

## [CRITICAL] C-2. Hybrid RAG resolves BM25 hits without the ACL filter — private chunks of other users leak into `/chat`

**Category:** Security (broken access control / excessive data exposure)
**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/ml/rag_service.py:118-144` (function accepts `access_filter`, never uses it), consumed at `:220` inside `_run_hybrid_search`.

### Problem

```python
async def _resolve_hash_to_doc(h: str, access_filter, ml_clients) -> LCDocument | None:
    ...
    scroll_filter=Filter(must=[FieldCondition(key="metadata.content_hash", match=MatchValue(value=h))]),
```

The BM25 index is **global** (built over all chunks; `bm25_updater`, `hybrid.py`), so sparse hits include chunks outside the caller's scope. Hashes not returned by the ACL-filtered dense arm are resolved by this unfiltered scroll and appended to candidates (`:213-222`) → rerank → LLM context → `sources` event to the client.

### Why it matters

The carefully built Qdrant ACL filter (`infrastructure/acl.py`, `build_qdrant_filter`) is applied only to the dense arm. Iterating distinctive vocabulary from another user's private document lets an authenticated user extract content through `sources` and answers, without any exact match requirement.

### Reproduction / scenario

User A (internal, no groups) asks a question containing a rare token that occurs only in user B's `internal_private` chunk. BM25 ranks it top; dense (ACL-filtered) search does not return it; `_resolve_hash_to_doc` fetches it by hash; it enters `sources` and the answer.

### Evidence

`rag_service.py:118` signature takes `access_filter`; `:125-132` the scroll filter contains only the hash condition; `:220` `doc = await _resolve_hash_to_doc(h, access_filter, ml_clients)`.

### Fix

Combine the filters: `Filter(must=[access_filter, hash_condition])`, or post-filter resolved docs with `is_in_search_scope(...)` (`access_control.py:180`). Add a chat-level test: private chunk of user B must never appear in A's candidates/sources.

### Tests

`tests/test_rag_logic.py` covers RRF merge only; no test covers ACL propagation through the hybrid path.

---

## [CRITICAL] C-3. Synchronous OCR / PDF analysis / page rendering run directly on the API event loop — one admin request freezes the entire API

**Category:** Async / availability
**Confidence:** CONFIRMED
**Location:** `server/app/presentation/api/routes/admin_quality.py:228` (`strategy.analyze(tmp_path)`), `:276-280` (`strategy.analyze` + `strategy.ocr_problem_units`), `:347, 355` (`diag_service._pdf.open`, `render_page_image`); implementation `server/app/application/services/pdf_diagnostic_service.py:122-180`; PaddleOCR inference `server/app/infrastructure/ml/ingestion.py:55-93`.

### Problem

`analyze()` (sync PyMuPDF over every page) and `ocr_problem_units()` (PaddleOCR torch inference per page) are called **directly in async route handlers without `asyncio.to_thread`**. The server runs a **single uvicorn process** (`presentation/cli/commands/runserver.py:26-36` — no `workers` parameter) with one event loop. Up to 50 MB PDFs are accepted (`pdf_diagnostic_service.py:77`).

### Why it matters

While one admin runs a dry-run OCR, **all** requests stall for seconds-to-minutes: chat SSE streams stall, health checks (`interval 30s, timeout 10s, retries 10`) start failing and Docker may restart the container mid-OCR — the exact "freeze the flagship RAG service with one request" scenario.

### Reproduction / scenario

1. Admin uploads a 40 MB scanned PDF to `POST /admin/documents/preview-ocr` with several problem pages.
2. Every concurrent request (chat, login, health) times out until OCR completes.

### Evidence

Direct call sites listed above; no `to_thread`/executor anywhere in `admin_quality.py` (grep verified). Contrast with `password_hasher.py:19-21`, which correctly uses `asyncio.to_thread`.

### Fix

`await asyncio.to_thread(strategy.analyze, tmp_path)` / `ocr_problem_units` / `render_page_image`; better — move dry-run OCR into the arq worker (it already has the OCR stack) with job status polling.

### Tests

No route-level tests exist at all (see Testing Gaps).

---

## [CRITICAL] C-4. arq default `job_timeout=300s` cancels long jobs past every `except Exception` → stuck documents, corrupted job state

**Category:** Concurrency / background processing
**Confidence:** CONFIRMED (code-verified; manifestation time depends on job length)
**Location:** `server/app/presentation/cli/commands/worker.py:58-67` (Worker built without `job_timeout`), `server/app/infrastructure/worker/tasks.py:44-86, 103-115, 132-144, 159-176, 285-292` (only `except Exception`), `:309-315` (`cron_recover_orphaned_jobs`, 15 min).

### Problem

arq's default per-job timeout is 300 s. `process_document` (OCR), `run_full_ingest`, `run_benchmark`, `run_sweep` routinely exceed it. On timeout arq cancels the task; `asyncio.CancelledError` derives from `BaseException`, so every `except Exception` in `tasks.py` is bypassed → `mark_failed` never runs.

Failure sequence:
- **Initial:** document 42 `processing`; job 11 `running` (`tasks.py:45-46`).
- **Operation A:** arq cancels job 11 at t+300 s mid-OCR → task dies silently, no DB write.
- **Operation B:** `cron_recover_orphaned_jobs` (every 15 min) marks job 11 `failed` **while the same document is never revisited** — there is no reconciler for documents stuck in `PROCESSING` (`outbox_dispatcher.reconcile_stuck_documents` handles only `INDEXING` with zero pending outbox entries, `outbox_dispatcher.py:141-163`).
- **Final:** document 42 remains `PROCESSING` forever (invisible in `list_documents` enrichment, un-searchable); job history shows `failed` with a misleading "Worker died" error.

Secondary effect: because task bodies swallow ordinary exceptions, arq's retry mechanism (`max_tries`) never triggers — the system is effectively **at-most-once** while the outbox design comments assume at-least-once.

Also: `cron_recover_orphaned_jobs` has no heartbeat — a legitimately long `run_full_ingest` (>15 min) is marked `failed` **while still executing**, and `mark_done` later overwrites → status flaps `running→failed→done` (`sqlalchemy_background_job_repository.py:124-139`, plain `UPDATE ... WHERE status='running'`).

### Evidence

`worker.py:58-67` (no `job_timeout`/`timeout` passed); `tasks.py:74, 109, 138, 170, 285` (`except Exception`); `outbox_dispatcher.py:141-163` (INDEXING-only reconcile).

### Fix

Per-function explicit timeouts (`arq.worker.function(..., timeout=...)` sized per job class), `except asyncio.CancelledError` handlers that persist `failed` + document failure state, a heartbeat column for orphan recovery, and a reconciler for `PROCESSING` documents older than N minutes.

### Tests

No worker-level tests; no test asserts a cancelled job leaves a recoverable state.

---

## [CRITICAL] C-5. BM25 sparse index is per-process: ingested/edited-in-worker documents are invisible to the API's hybrid search until API restart

**Category:** Data consistency / architecture
**Confidence:** CONFIRMED (single-instance verified by code; two processes are part of the standard compose stack)
**Location:** `server/app/infrastructure/ml/client_registry.py:86-92` (`bm25_index()` lazy-loaded once, `self._bm25_loaded = True`), `:118-121` (`invalidate_bm25` called only on the `hybrid_enabled` config event — `composition/container.py:133-135`), `server/app/infrastructure/ml/bm25_updater.py` (in-memory mutations), `server/app/infrastructure/worker/tasks.py:318-346` (cron rebuild persists to S3 only).

### Problem

The API process loads the BM25 index **once** from S3 and mutates it only for in-process events (chunk edit/add/delete via `ChunkService`/`DocumentService` call `bm25_add/replace/remove`). The **worker** process (which performs ingestion, single-file ingest, daily rebuild) maintains its own copy and saves to S3 — the API process never reloads it. There is no pub/sub, no NOTIFY, no TTL invalidation for BM25.

Failure sequence:
- **Initial:** API BM25 index = corpus at T0. New document ingested via `POST /ingest/file` → processed in worker at T1; worker saves new index to S3.
- **Operation A:** user asks a question at T2 — dense arm finds the new doc; sparse arm doesn't (index stale).
- **Final:** hybrid fusion silently under-ranks new documents; BM25-driven features (exact-ref boost, hybrid fetch) degrade until API restart.

### Evidence

`client_registry.py:86-92` loads once; grep for `invalidate_bm25` shows the only caller is the `hybrid_enabled` config handler. No reload notification exists anywhere (grep verified).

### Fix

After `cron_bm25_rebuild` / ingestion BM25 save, publish an invalidation event (existing Redis or `config_changed`-style NOTIFY infra can be reused) → API reloads; or move BM25 to a shared store (Redis) with versioned keys.

### Tests

`tests/test_bm25_incremental.py` tests in-process add/replace only — cross-process propagation is untested (and unimplemented).

---

# High Findings

## [HIGH] H-1. Synchronous S3 `delete_file`/`rename_file` (boto3, no await) inside open DB transactions — event-loop block + dangling references + S3/DB divergence

**Category:** Transactions / async / partial failure
**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/storage/file_storage.py:169-181` (sync boto3, no `to_thread`, no boto3 `Config` timeouts); call sites inside open UoWs: `server/app/application/services/document_service.py:163` (conflict replace), `:406` (delete), `:447` (rename); `server/app/application/services/document_processor.py:285` (replace path).

### Problem

- The calls block the API event loop for up to botocore's default read timeout (~60 s) **while holding a Postgres transaction and, in the upload path, a `FOR UPDATE` row lock** (`find_active_slot(for_update=True)`, `document_repository.py:100-125`).
- Partial failure: `rename_file` = copy + delete; if copy succeeds and delete fails, the exception propagates → transaction rolls back → DB still points at `old_key`, but the object was deleted → dangling `source_path`.
- Delete-then-rollback: `delete_document` deletes the S3 object (line 406) before the DB delete (409); a failure in the DB delete leaves DB row → deleted object.

### Reproduction / scenario

Rename doc 7 (same owner) while MinIO is degraded: copy OK, delete raises → `BusinessRuleViolation`-style 500, transaction rolled back, document's `source_path` now points to a nonexistent S3 object → the document becomes un-downloadable and un-reindexable.

### Fix

Perform storage mutations **after** the DB transaction commits (compensating action on failure), or emit them via the existing outbox; wrap boto3 with explicit `Config(connect_timeout=..., read_timeout=..., retries=...)` and always call through `asyncio.to_thread`.

### Tests

No test simulates storage failure between DB mutation and commit.

---

## [HIGH] H-2. Synchronous Qdrant `client.upsert` / `get_collections` on the API event loop — the outbox dispatcher (which runs inside the API scheduler) blocks the loop on every batch

**Category:** Async / performance
**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/qdrant_ops.py:156` (`create_qdrant_client()` per call), `:184-189`, `:203-208` (sync `client.upsert` inside `async def upload_to_qdrant`), `:37, 57, 64, 75` (sync `get_collection(s)`); invoked via `QdrantVectorStoreRepository.upload_documents` → `OutboxDispatcher._apply_upsert` (`outbox_dispatcher.py:125-139`) → APScheduler job `_periodic_outbox_dispatch` **in the API process** (`scheduler.py:84-90`, every 30 s), and via NOTIFY-triggered dispatch.

### Problem

Everything else in the Qdrant repository correctly wraps calls in `asyncio.to_thread` (`qdrant_vector_store_repository.py`), but `qdrant_ops` performs 500-point, 1024-dim upserts synchronously on the loop. A large document's outbox entry produces many batches → repeated multi-100ms freezes → SSE chat stutter, rate-limiter latency, health-check flakiness.

### Reproduction / scenario

Ingest 1,000 chunks (≈2 upsert batches per entry, several entries). During dispatch each `upsert` blocks the loop; concurrent `/chat` requests queue behind it.

### Fix

`await asyncio.to_thread(client.upsert, ...)`; reuse one Qdrant client instead of constructing per call (`qdrant_ops.py:156`).

### Tests

None (no async-blocking regression tests exist).

---

## [HIGH] H-3. Ingestion registry is committed before documents/chunks/outbox are synced — crashed sync permanently removes files from the pipeline

**Category:** Transactions / data loss
**Confidence:** CONFIRMED
**Location:** full ingest: `server/app/infrastructure/services/ingestion_service.py:241` (`_build_registry_entries`, own committed UoW) **before** `:245` (`_sync_documents_to_db`); single file: `:420-426` (`_registry_upsert`) **before** `:427` (`_sync_documents_to_db`); skip logic `:399-401, 533-537` (`_registry_is_indexed` → `CACHED` → skip).

### Problem

The registry row says "indexed" while the document/chunks/outbox may not exist. If the process crashes or `_sync_documents_to_db` raises, the next incremental run reports `CACHED` and **silently never indexes the file** — catalog loss without any error.

### Reproduction / scenario

1. `POST /ingest/file` for `report.pdf`; parsing/embedding succeeds, registry row committed.
2. DB blip during `_sync_documents_to_db` → exception → job marked failed.
3. Re-run `POST /ingest/file` → `_registry_is_indexed` → `CACHED` → nothing indexed, HTTP 200-ish "started".

### Fix

Upsert the registry **in the same transaction** as `_sync_documents_to_db` (the function already accepts `_existing_uow` for chunks), or mark registry rows with a `synced` flag and only set it after the sync transaction commits.

### Tests

No test covers registry/sync crash ordering.

---

## [HIGH] H-4. Whole-corpus ingest runs in ONE database transaction with a direct Qdrant write inside — minutes-long lock window and outbox-pattern violation

**Category:** Transactions / concurrency
**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/services/ingestion_service.py:290` (`async with self._uow_factory.create(master=True) as uow` spanning the per-document loop over the entire registry), `:357` (`await self._vector_store.set_document_id_by_source(src, doc_id)` — direct Qdrant write inside the transaction).

### Problem

- One UoW spans every document of the corpus; row locks (and WAL) are held for the entire ingest; any failure rolls back **everything**.
- `set_document_id_by_source` writes to Qdrant before commit: a rollback leaves Qdrant patched with a `document_id` that was never committed — precisely what the outbox exists to prevent (the same module otherwise uses `process_chunks` with `_existing_uow=uow` correctly at `:343-354`).

### Reproduction / scenario

500-file ingest; Qdrant hangs on file 499 → 15-minute transaction → connection-pool pressure, replication lag, and a full rollback; or Qdrant call succeeds on file N, DB rolls back at N+1 → Qdrant points chunks at document IDs that no longer exist.

### Fix

Per-document transaction; route `set_document_id_by_source` through the outbox (it is a vector-store mutation).

### Tests

`tests/test_saga_outbox.py` covers outbox mechanics but not the ingest transaction boundary.

---

## [HIGH] H-5. `DocumentProcessor.process` opens a nested transaction (missing `_existing_uow`) — DONE→INDEXING status regression and duplicated corpus on replace failure

**Category:** Transactions / consistency
**Confidence:** CONFIRMED
**Location:** `server/app/application/services/document_processor.py:243` (outer UoW) → `:261-271` (`process_chunks(...)` **without** `_existing_uow`, cf. the pipeline's own contract `document_pipeline.py:98, 112-113, 118-132`); `:274-286` replace handling inside the outer tx; `:289-295` post-commit-warning update.

### Problem

`process_chunks` with `_existing_uow=None` commits chunks + outbox + `INDEXING` in an **inner** transaction while the outer transaction is still open. Sequence:

- **Initial:** doc 42 `processing`; inner tx commits chunks + outbox + `INDEXING`.
- **Operation B:** outbox dispatcher applies UPSERT to Qdrant → `count_by_document` pending == 0 → sets doc 42 **DONE** (`outbox_dispatcher.py:75-92`).
- **Operation A:** outer tx commits `update_status(42, INDEXING, warning=...)` **after** B.
- **Final:** doc 42 shows `INDEXING` although fully indexed (only the 60-s reconciler re-flips it to DONE); `indexed_at` regresses.

Worse partial-failure path: if anything in the outer tx after the inner commit fails (e.g. the **synchronous S3 delete at `:285`** raises), the outer tx rolls back → inner-committed chunks/outbox survive, but the **old document is not deleted** → both old and new documents exist → duplicate searchable content in Qdrant.

### Evidence

`document_pipeline.py:118-132` explicitly supports `_existing_uow` "чтобы chunks попали в ту же транзакцию"; `ingestion_service.py:353` passes it; `document_processor.py:261` does not.

### Fix

Pass `_existing_uow=uow` at `document_processor.py:261`; make the dispatcher's DONE transition conditional (`UPDATE ... WHERE status='indexing'`); move the S3 delete out of the tx.

### Tests

`tests/test_saga_outbox.py::test_document_goes_to_done_after_qdrant_success` exists, but the nested-tx path via `DocumentProcessor.process` is not covered.

---

## [HIGH] H-6. No rate limiting on `POST /auth/login` and rate-limit identifier trusts `X-Forwarded-For` — brute force with trivial bypass

**Category:** Security (authentication)
**Confidence:** CONFIRMED
**Location:** `server/app/presentation/api/routes/auth.py:18-22` (no RateLimiter); `server/app/presentation/api/rate_limits.py:32-39` (identifier takes the **first** entry of `X-Forwarded-For`); server published directly on `0.0.0.0:8001` (`docker-compose.yml:247-248`); client nginx appends with `$proxy_add_x_forwarded_for` (client Dockerfile nginx config) — attacker-controlled values survive.

### Problem

1. Unlimited password guessing against any account (JWT is then valid 24 h).
2. Even on limited endpoints, the IP fallback key is client-supplied: send a fresh `X-Forwarded-For: 1.2.3.N` per request → per-IP limits never trip.

### Evidence

`rate_limits.py:32-34`: `forwarded = request.headers.get("X-Forwarded-For"); ip = forwarded.split(",")[0]`. Only three limiters exist (`chat_rate_limit`, `upload_rate_limit`, `ingest_rate_limit`) — none on login or `/search/exact`.

### Fix

Rate-limit `/auth/login` per-IP + per-email (Redis counter), take the client IP from the **last** untrusted-hop-stripped value or socket `request.client.host` only (uvicorn `--proxy-headers` with `forwarded_allow_ips` set to the nginx network), add failed-attempt lockout.

### Tests

No rate-limit tests; no brute-force tests.

---

## [HIGH] H-7. Fire-and-forget enqueue after separate commit → permanently `pending` jobs; per-call Redis pool churn

**Category:** Transactions / reliability
**Confidence:** CONFIRMED
**Location:** `server/app/presentation/api/helpers.py:114-126` (upload: `create_job` commits → `enqueue_fn` may fail), `server/app/presentation/api/routes/ingest.py:44-57, 79-89`, `benchmark_admin.py:233-241`; `server/app/application/services/job_service.py:19-29`; `server/app/infrastructure/worker/queue.py:19-28` (`create_pool` + `close` per enqueue, no `_job_id` → no dedup, see H-8).

### Problem

The job row is committed in its own UoW; the subsequent Redis `enqueue_job` can fail (Redis blip). `cron_recover_orphaned_jobs` reaps only `running` jobs — a `pending` job that was never enqueued stays pending forever; the document is never processed and no retry occurs.

### Reproduction / scenario

Upload a document while Redis is restarting: `create_job` commits, `enqueue` raises → 500 to the user, document row exists with `pending` job, nothing will ever process it.

### Fix

Re-enqueue pending jobs periodically (background sweep of `pending` jobs older than X), or enqueue-then-create-row with compensation; keep a shared arq pool instead of per-call pools.

---

## [HIGH] H-8. Background jobs have no idempotency key — double-click / retry causes duplicate full ingestion and duplicate chunks/vectors

**Category:** Concurrency / idempotency
**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/worker/queue.py:25` (`enqueue_job(...)` without `_job_id`); check-then-act guard `server/app/infrastructure/services/ingestion_service.py:399-401, 533-537` (registry lookup is not atomic and races between two workers); same for `POST /documents` via `helpers.py:114-126` (protected only by the partial unique index + 409, see M-11).

### Problem

Two rapid `POST /ingest` calls (limit: 10/min/admin) → two `run_full_ingest` jobs run concurrently (`max_jobs=4`) → both pass `_registry_is_indexed` → both parse & index → **duplicated chunks (new chunk ids), duplicated outbox UPSERT entries, duplicated Qdrant points**, registry upsert last-writer-wins.

### Reproduction / scenario

- Initial: no registry entry for `report.pdf`.
- Request A and Request B both enqueue `run_full_ingest`; both workers run `_load_documents` before either commits the registry.
- Final: two copies of every chunk of the file in Postgres and Qdrant; search returns duplicates.

### Fix

Deterministic arq `_job_id` (e.g. `ingest:{path-hash}`, `process_doc:{job_id}`) so arq deduplicates; add `UNIQUE` on `ingestion_registry.filename` semantics check **inside** the sync transaction (`INSERT ... ON CONFLICT DO NOTHING`).

### Tests

No concurrency tests for double enqueue.

---

## [HIGH] H-9. Outbox stuck-recovery timeout (5 min) is shorter than worst-case apply (600 s embedding timeout); `recover_stuck` is SELECT-then-update with a lost-update window

**Category:** Concurrency
**Confidence:** CONFIRMED (code-verified; duplicate apply requires a large batch / slow embedding)
**Location:** `server/app/infrastructure/outbox_dispatcher.py:20-22` (`_STUCK_TIMEOUT_MINUTES = 5`), `:56-70`; `server/app/infrastructure/repositories/sqlalchemy_vector_outbox_repository.py:82-96` (SELECT → ORM mutate → flush); embedding timeout 600 s (`infrastructure/ml/tei_clients.py:20`).

### Problem

- A dispatcher still working on a huge `upsert_chunks` entry (many embed batches × up to 600 s) is considered "stuck" after 5 min → another tick re-claims it → **two concurrent appliers**; both eventually call `mark_done`/`mark_failed` — last write wins (attempts/backoff can be lost). Upserts themselves are idempotent (deterministic chunk ids) so no vector corruption — the cost is duplicate embedding spend and bookkeeping races.
- `recover_stuck` re-reads rows and flips status without a conditional `UPDATE`: if the original worker committed `done` between the reaper's SELECT and flush, `done` is overwritten back to `pending` and re-applied later.

### Fix

Single-statement conditional recovery `UPDATE ... SET status='pending' WHERE status='in_progress' AND locked_at < :cutoff`; raise the stuck timeout above the max plausible apply duration; consider lock renewal.

### Tests

None for the double-claim race.

---

## [HIGH] H-10. Conversation row lock held across an external LLM call (rolling summary)

**Category:** Transactions / concurrency
**Confidence:** CONFIRMED
**Location:** `server/app/application/services/chat_service.py:136-148`.

### Problem

```python
async with self._uow_factory.create(master=True) as uow:
    conv_model = await uow.conversations.get_for_update(conv_id)   # FOR UPDATE
    ...
    new_summary = await updater.update(existing, recent_turns)      # external LLM call
    conv_model.summary = new_summary
    await uow.conversations.save(conv_model)
```

The commit happens at context exit → the `FOR UPDATE` lock on the conversation row is held for the full LLM latency (seconds to minutes on a slow Ollama). Any concurrent chat on the same conversation blocks on the row (or times out).

### Reproduction / scenario

Two users chat in the same conversation concurrently (or a user sends two rapid messages); rolling summary triggers on the first; the second message's UoW-2 save blocks until the summary LLM responds.

### Fix

Compute the summary **outside** the transaction (read summary → LLM → short `UPDATE ... WHERE` at the end), or skip `FOR UPDATE` (single-writer per conversation is not required for a last-write-wins summary).

---

## [HIGH] H-11. Sweep Phase B is dead code in production: `get_setting("rag.retriever_top_k")` raises `AttributeError`

**Category:** Bug (business logic)
**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/ml/sweep_engine.py:484` vs `:455-468` (`_build_overrides` maps to bare key `retriever_top_k`), `server/app/config.py:21-30` (`get_setting` falls back to `getattr(settings, key)`).

### Problem

`get_setting("rag.retriever_top_k")` looks up `"rag.retriever_top_k"` in the override context (which contains `"retriever_top_k"`) → miss → `getattr(settings, "rag.retriever_top_k")` → `AttributeError`. The LLM-judged phase of **every** sweep fails; the sweep is marked failed after the retrieval metrics were computed.

### Fix

Use the bare key `"retriever_top_k"` (consistent with `_build_overrides`), or honor both.

### Tests

`run_sweep` has no test that exercises `_run_full_benchmark` (the fakes bypass it).

---

## [HIGH] H-12. Group members can delete the entire shared document but cannot edit a single chunk — destructive privilege asymmetry

**Category:** Security / policy
**Confidence:** CONFIRMED
**Location:** `server/app/domain/entities/document.py:66-80` (`can_be_deleted_by` grants any `internal_group` member delete) vs `:82-93` (`can_edit_chunks` denies non-admin for group docs); used in `document_service.py:389-390` with a misleading message "Can only delete your own documents".

### Problem

Any group member can destroy the shared document (DB cascade + Qdrant deletion via outbox + S3 delete) — a strictly more destructive action than the chunk editing they are denied. If this is intended collaboration, the error message and asymmetry are wrong; if not, it is an unintended privilege grant.

### Fix

Pick one policy: either members can edit chunks, or only owners/admins can delete group documents.

---

# Medium Findings

## [MEDIUM] M-1. Answer cache scope excludes `user_id` — cross-user answers when the admin toggles `cache_enabled`

**Confidence:** CONFIRMED (code flaw; gated by `cache_enabled`, default `False` but runtime-toggleable via `PUT /admin/config/{key}`)
**Location:** `server/app/infrastructure/ml/answer_cache.py:25-32` (`scope = f"{user_kind}:{sorted(group_ids)}"`), cache hit path `rag_service.py:640-646`.

All client-kind users share scope `client:[]`; all group-less internal users share `internal:[]`. Cached `answer` + `sources` (which may reference user A's private docs) are served to user B asking the identical question. Also: `find_cached_answer` comment says "don't extend TTL on hit" but the code passes `ex=CACHE_TTL_SECONDS` (`:64-67`) — TTL is reset on every hit (comment/code contradiction).

**Fix:** include `user_id` in the scope hash; fix or remove the TTL comment.

---

## [MEDIUM] M-2. Config seeding is check-then-insert without `ON CONFLICT`, and the unique `(key, domain_key)` index does not dedupe NULLs — concurrent startups silently create duplicate rows → every config read/update 500s

**Confidence:** CONFIRMED (single-statement-level; manifestation requires overlapping starts, e.g. `--reload`, scaled replicas)
**Location:** `server/app/infrastructure/initialization.py:297-313, 328-360` (read-all → check → plain `save`/INSERT); `config_parameter_repository.py:63-76` (plain INSERT); unique index `ux_config_parameters_key_domain` (migration `x1a2b3c4d5e6`); all ~60 seeded params have `domain_key = NULL` — Postgres treats NULLs as distinct.

Duplicate rows make `get_by_key`-style `scalar_one_or_none` raise `MultipleResultsFound` → `ConfigService.update_parameter` (`config_service.py:39`) and every config update 500 forever until manual cleanup. Failures are swallowed at startup (`initialization.py:324-325` logs a warning and continues).

**Fix:** `INSERT ... ON CONFLICT (key, domain_key) DO UPDATE` (or `NULLS NOT DISTINCT` on PG15+) + `IntegrityError` handling.

---

## [MEDIUM] M-3. Group management raises raw `IntegrityError` → 500 on duplicate member, nonexistent group, duplicate group name

**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/repositories/sqlalchemy_group_repository.py:16-20` (create — unique `name`), `:40-44` (add_user — composite PK duplicate / FK to missing group), `server/app/application/services/group_service.py:25-44` (no existence check, no error mapping); routes `groups.py:21-62` (no exception handling).

`POST /groups/{id}/members` with an already-member user or a nonexistent group → HTTP 500. Should be 404/409.

---

## [MEDIUM] M-4. Unauthenticated `/metrics` endpoint + unbounded Prometheus label cardinality in the custom middleware

**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/ml/metrics_middleware.py:13-19` (Instrumentator exposes `/metrics` with no auth; server also published directly on `0.0.0.0:8001`); `server/app/presentation/api/middleware/metrics.py:12-17` (`HTTP_REQUESTS_TOTAL.labels(handler=request.url.path, ...)` — **raw path**, including ids: `/documents/123`, `/admin/documents/preview/{id}/index`).

An attacker can generate unique paths (or unique object ids) to grow the Prometheus registry without bound → memory growth on the API process (label-value explosion). `/metrics` discloses infrastructure details (stage, hostnames in job labels, request rates) to anyone who can reach :8001.

**Fix:** restrict `/metrics` (auth or bind-internal); use route templates (`request.scope["route"].path`) or cap label values in the custom middleware.

---

## [MEDIUM] M-5. Blocking network calls in 30-second scheduler ticks (API event loop): Qdrant `get_collection` and potential full BM25 S3 load + rebuild

**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/ml/metrics.py:309-328` (`_collect_qdrant_metrics` uses sync client; `_collect_bm25_metrics` → `ml_clients.bm25_index()` → lazy load from S3 (sync boto3 `get_object`, `file_storage.py:183-190`) **plus full-corpus tokenize/stem CPU rebuild on first access**); also runs at startup (`main.py:115`). Same class of issue as C-3/H-2 but periodic.

**Fix:** `to_thread` around all sync calls; pre-load the BM25 index in the lifespan explicitly (or serve BM25 size from a cached counter).

---

## [MEDIUM] M-6. `GET /documents` has no pagination; `list_visible`/`list_all` unbounded; `reconcile_stuck_documents` is a per-minute N+1 over all documents

**Confidence:** CONFIRMED
**Location:** `server/app/presentation/api/routes/documents.py:107-114` (no limit/offset); `sqlalchemy_document_repository.py:187-194` (no LIMIT); `outbox_dispatcher.py:147-163` (`list_all()` + `count_by_document` per doc every 60 s — 100k docs → 100k queries/min); related N+1: `document_service.py:297-302` (outbox count per `indexing` doc on every list call).

**Fix:** paginate `GET /documents` (schema supports it elsewhere); make reconcile use a single `WHERE status='indexing' AND NOT EXISTS(pending outbox)` query.

---

## [MEDIUM] M-7. `conversations.list_by_user` aggregates the entire messages table on every page request

**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/repositories/sqlalchemy_conversation_repository.py:78-104` — subqueries compute `min(id) WHERE role='user' GROUP BY conversation_id` and `count()` **over all messages of all users**, then outer-join. Cost grows O(all messages) regardless of page size.

**Fix:** filter subqueries by the user's conversation ids (or LATERAL join).

---

## [MEDIUM] M-8. `find_duplicate_by_hash` ignores the `content_hash` index — loads ALL chunks of a document and re-hashes in Python on every chunk edit/add

**Confidence:** CONFIRMED
**Location:** `server/app/infrastructure/repositories/sqlalchemy_chunk_repository.py:288-305` (selects all chunk rows, `hashlib.sha256(...)` loop), index `idx_chunks_content_hash` exists (migration `p5q6r7s8t9u0`), called from `chunk_service.py:129, 226`.

**Fix:** `WHERE document_id = :d AND content_hash = :hash` (+ exclude id). Note: the ORM model does not even declare `content_hash` index (drift — see D-3).

---

## [MEDIUM] M-9. Rename/delete outbox payloads are silently truncated at 10,000 chunks; full-content payloads are written inside transactions

**Confidence:** CONFIRMED (trigger requires >10k-chunk documents — possible: chunk_size can be lowered to 100 via admin config)
**Location:** `server/app/application/services/document_service.py:473` (`(await uow.chunks.list_for_document(document_id, limit=10000))[0]`), `:104` (BM25 removal same limit); payload of every chunk's full content is embedded in the `vector_store_outbox.payload` JSON column inside the rename transaction (memory + WAL blow-up for big documents).

**Fix:** split payloads into batches keyed by document; or use a `delete_chunks(doc_id)` + reindex flow; enforce a max chunk count per document.

---

## [MEDIUM] M-10. Raw, unredacted user questions are written to audit logs; SSE error events leak raw exception text

**Confidence:** CONFIRMED
**Location:** PII: `server/app/presentation/api/routes/chat.py:43-50, 101-108` (`log_action("chat", details={"question": req.question[:...]})` — redaction happens only later inside `ChatService._redact_pii`, `chat_service.py:112-123`, and only in DB storage). Exception leak: `chat.py:77-78` (`yield f"event: error\ndata: {json.dumps({'error': str(e)})}"`) — internal exception details (DB errors, paths) go to end users. Same pattern at `benchmark_admin.py:302-304` (admin-only there).

**Fix:** redact before `log_action`; emit a generic message in SSE and log details server-side.

---

## [MEDIUM] M-11. Public/group (NULL-owner) documents bypass the duplicate-slot guard — NULLs-distinct unique index + lock that locks nothing

**Confidence:** CONFIRMED
**Location:** unique index `ux_documents_active_slot (owner_id, filename) WHERE status IN (...)` (`models.py:100-106`, migration `a1b2c3d4e5f6:145-151`, widened `u3v4w5x6y7z8`); `compute_owner_and_group` returns `owner=None` for `internal_public`/`internal_group` (`access_control.py:140-143`); `find_active_slot(for_update=True)` locks nothing when no row exists (PG has no gap locks at READ COMMITTED).

Two concurrent uploads of the same public filename both insert (`NULL != NULL` in unique indexes) → two active documents with the same filename → duplicated chunks/vectors and nondeterministic `find_active_slot` (latest by creation). Same-owner user uploads are protected by the index (mapped to 409 at `document_service.py:228-235`) — public/group paths are not.

**Fix:** expression unique index on `(COALESCE(owner_id, 0), COALESCE(group_id, 0), filename) WHERE status IN (...)`, or a Postgres advisory lock keyed by (visibility, filename) around upload.

---

## [MEDIUM] M-12. SSE sweep progress stream terminates on the first idle timeout — UI stops updating while the sweep continues

**Confidence:** CONFIRMED
**Location:** `server/app/presentation/api/routes/benchmark_admin.py:275-304` — `except TimeoutError` is outside the `while True` loop: after the first 30 s without a message the generator yields one heartbeat and **exits**; the `done` event is missed.

**Fix:** catch the timeout inside the loop (`continue` after emitting heartbeat).

---

## [MEDIUM] M-13. No cross-instance/cross-process locks for scheduler and cron jobs

**Confidence:** HIGHLY LIKELY (single instance today; compose runs one API + one worker, but the code has no guard for scale-out)
**Location:** `server/app/infrastructure/scheduler.py:59-67` (`max_instances=1` is per-process), `server/app/presentation/cli/commands/worker.py:49-53` (arq cron runs in every worker process).

Scaling the worker to 2 replicas runs two concurrent BM25 rebuilds (both `get_all_contents()` full-table reads → last-writer-wins to S3 → different instances load different snapshots) and two orphan-reapers double-marking. Scaling the API runs two outbox dispatchers (safe via SKIP LOCKED, but duplicated work).

**Fix:** pg advisory locks (or Redis `SET NX PX`) around cron bodies; designate a single dispatcher owner.

---

## [MEDIUM] M-14. Sweep double-submit; `cancel_sweep` only flips DB status — the running engine never re-reads it

**Confidence:** CONFIRMED (double-submit), HIGHLY LIKELY (cancel no-op — `cancel_sweep` at `benchmark_admin.py:313-320` calls `service.cancel` which only updates status; `SweepEngine.run_sweep` has no status polling loop; grep verified)
**Location:** `benchmark_admin.py:226-241` (no `pending/running` guard).

Two clicks → two sweeps → double LLM spend. Cancel is cosmetic until the job ends by itself.

**Fix:** guard on active sweep status; cooperative cancellation flag checked between configs.

---

## [MEDIUM] M-15. Read-replica splitting is unsafe for the auth path and freshly-written data

**Confidence:** HIGHLY LIKELY (single-node mode today routes reads to master, `database.py:253-254, 277`; becomes real the moment `DB_SLAVE_HOSTS` is set)
**Location:** `database.py:382-394` (round-robin slaves), `uow_factory.py:53` (`master=False` default), consumers: JWT auth per request (`auth_dependencies.py:35` → `get_user_by_id`, non-master), login (`auth_service.py:35`), API-key validation (`auth_service.py:115-116` — a newly issued key may 401 until the replica catches up), act lookup (`act_versioning_service.py:44`), config resync (`postgres_config_listener.py:102`).

Also missing: `pool_pre_ping`, `pool_recycle`, `statement_timeout` (`database.py:75-82`); pool sizing `100+20` × (server + worker) = 240 vs Postgres `max_connections` (compose conf) — headroom of ~60 across **all** other clients.

**Fix:** route auth to master (or pin read-your-writes), add pool hardening, document replica lag policy.

---

## [MEDIUM] M-16. Uploads are fully buffered into memory before the size check

**Confidence:** CONFIRMED
**Location:** `server/app/presentation/api/routes/documents.py:74-76` (`data = await file.read()` then `len(data) > max_bytes`). A client (reaching :8001 directly, bypassing nginx limits) can upload arbitrarily large multipart bodies → memory spikes × concurrent requests. The client nginx does not help (see M-18).

**Fix:** stream-read with a size cap (`UploadFile` chunked read) or enforce `client_max_body_size` at every hop.

---

## [MEDIUM] M-17. `bootstrap_admin` and seeding swallow all exceptions at startup — server starts without an admin

**Confidence:** CONFIRMED
**Location:** `initialization.py:286-287` (`except Exception as e: logger.warning(...)`), `:324-325`, `:359-360`, `:383-384`. A transient DB error during first boot leaves the system with **no admin user** and default config, while the server reports healthy.

**Fix:** fail fast (raise) for first-boot bootstrap, or add a startup health check that reports "admin missing".

---

## [MEDIUM] M-18. Client nginx lacks `client_max_body_size` — UI uploads >1 MB fail with 413

**Confidence:** CONFIRMED (nginx default 1 MB; the generated config sets no limit)
**Location:** client Dockerfile serve stage — the `printf`-generated `/etc/nginx/conf.d/default.conf` sets proxy timeouts but no `client_max_body_size`; `max_upload_size_mb=50` in backend settings. All uploads via the web UI (`VITE_API_URL=/api`) >1 MB are rejected by nginx before reaching FastAPI.

**Fix:** `client_max_body_size 64m;` (aligned with backend limit) in the generated config.

---

# Low Findings

## [LOW] L-1. JWT: no `iat`/`jti`, no revocation list; malformed-but-validly-signed token can 500
`jwt_provider.py:17-23` — only `sub`, `role`, `exp`. Revocation = deactivation (checked per request, good). `auth_dependencies.py:35` `int(payload["sub"])` raises `KeyError/ValueError` on a crafted token signed with the app secret → 500 instead of 401. Impact minimal (requires the secret), but map to `AuthenticationError`.

## [LOW] L-2. Password policy absent; bcrypt 72-byte truncation
`CreateUserRequest`/`LoginRequest` accept any `password: str` (`schemas.py:145-147, 236-241`); bcrypt silently truncates >72 bytes (`password_hasher.py`). Admin-created accounts may have 1-char passwords. Add min length at creation (admin surface) and reject >72-byte passwords.

## [LOW] L-3. Login account enumeration via timing
`auth_service.py:36-40` — bcrypt verification runs only when the user exists → response-time oracle. Same generic message (good), but timing differs. Fix: run a dummy verify when user is missing.

## [LOW] L-4. 404 vs 403/409 object-existence oracles
Nonexistent → 404 (`conversation_service.py:27`, `document_service.py:341`), no-access → 403/409 (`conversation_service.py:29`, `document_service.py:82` mapped via `exception_handlers.py:38-40`). Sequential integer ids make enumeration easy. Use uniform 404 for no-access.

## [LOW] L-5. Weak default credentials shipped in `.env.example` and compose
`.env.example`: `JWT_SECRET_KEY=change-me-in-production`, `ADMIN_PASSWORD=admin`, `ALLOWED_ORIGINS=*`; `docker-compose.yml:100, 167-169` hardcodes `ragpassword` and Redis `--requirepass password` (healthcheck embeds it too). The prod guard (`config.py:246-253`) only enforces JWT secret and file_backend when `STAGE=prod`; `stage` defaults to `development`. Add checks for admin password / redis password / CORS in prod mode; never ship real defaults in compose.

## [LOW] L-6. MinIO console published on `0.0.0.0:9001`; API port published on `0.0.0.0:8001`
`docker-compose.yml:413-425, 247-248`. Console (root creds) exposed on all interfaces; FastAPI bypasses the client nginx. Bind console/API to 127.0.0.1 or the docker network; access the UI only through the nginx proxy.

## [LOW] L-7. `POST /upload` (ingest) builds S3 keys from raw multipart filenames without `_validate_s3_key`
`ingestion_service.py:441-449` (`key = prefix + f.filename`, no sanitization; contrast `document_service._storage_key` which uses `Path(filename).name`). S3 keys are flat strings so there is no true traversal, but keys can escape the `docs/` prefix and collide with reserved objects (e.g. the BM25 index object) or contain `../` noise. Admin-only; sanitize.

## [LOW] L-8. `touch_api_key_last_used` performs a DB write on every API-key cache miss
`auth_dependencies.py:51` + `auth_service.py:128-131` — every request with a cold key opens a master UoW to update `last_used_at`. Under a client-traffic burst with a cold Redis this multiplies write load. Throttle (update at most once per minute/key).

## [LOW] L-9. `user.create` race maps to 500 instead of 409
`auth_service.py:61-65` does `get_by_email` → `save` without catching `IntegrityError` (the unique constraint exists, so no corruption — just an unhandled 500; contrast `document_service.upload:228-235` which maps it). Wrap and map to `BusinessRuleViolation`.

## [LOW] L-10. Regulatory acts: no `UNIQUE (act_type, act_number)`, no partial unique for `is_current`, find-or-create reads replicas
`models.py:185-216`; `act_versioning_service.py:44-57` (`create()` non-master read → save master). Concurrent versioned uploads (or replica lag) can create duplicate acts and two `is_current=true` versions; `unset_current` then clears only one act's versions. Add constraints + master read for the lookup.

## [LOW] L-11. `chunks` table has no `UNIQUE (document_id, chunk_index)`; manual chunk append is SELECT max → INSERT
`models.py:219-249` (no constraint); `chunk_service.py:228-229` (`get_max_chunk_index` → `next_index`) + `insert_one` (`chunk_repository.py:113-140`). Two concurrent `add_chunk` calls can produce duplicate `chunk_index` (ordering/citation drift). Add the unique constraint; retry on conflict.

## [LOW] L-12. Messages ordering is fragile: `ORDER BY creation_date DESC` without `id` tiebreaker; no index on `messages.creation_date`
`sqlalchemy_message_repository.py:27-33`. Two messages saved in one transaction can share a timestamp (Python-side defaults) → history order non-deterministic on ties. Order by `(creation_date DESC, id DESC)` and add a composite index.

## [LOW] L-13. `increment_evaluated` is read-modify-write (lost updates)
`sqlalchemy_benchmark_sweep_repository.py:48-54` (`orm.evaluated_configs += 1` after SELECT). Use `UPDATE ... SET evaluated_configs = evaluated_configs + 1`.

## [LOW] L-14. ORM ↔ migration drift
`models.py:352-362` lacks the partial `idx_outbox_dispatch (status, next_attempt_at) WHERE status IN ('pending','failed')` (created by `t2u3v4w5x6y7:62-67`); `ChunkModel` lacks `ix_chunks_visibility/owner_id/group_id` (restored via raw SQL in `r7s8t9u0v1w2:20-22`) and `idx_chunks_content_hash`. `create_all`-built schemas (tests) differ from production. Keep the metadata authoritative or exclude drift-prone indexes from ORM.

## [LOW] L-15. Naive/aware timestamp mix
`DocumentModel.indexed_at`, `ApiKeyModel.revoked_at/last_used_at`, jobs `started_at/finished_at`, act dates are `DateTime()` (naive, migrations `a1b2c3d4e5f6:127, 163`) while writers use `datetime.now(tz=UTC)` (`document_repository.py:83`, `api_key_repository.py:53, 87`). Session timezone is pinned to `Europe/Minsk` (`database.py:81`) regardless of `settings.timezone` (default UTC). Inconsistent tz semantics across columns; standardize on `DateTime(timezone=True)`.

## [LOW] L-16. `rag_service.py:438` dead condition
`use_llm = settings.llm_provider == LLMProvider.OPENROUTER or True` — always True; the non-LLM branch (`:454-458`) is unreachable. Remove or fix intent.

## [LOW] L-17. Request-ID contextvar set without reset inside RAG stream
`rag_service.py:620-623` — `request_id_ctx.set(req_id)` without storing the token; combined with chat route's own set/reset (`chat.py:41, 85-86, 99, 125-126`) the middleware value may be overwritten for the remainder of the request task. Cosmetic tracing inconsistency; use token/reset.

## [LOW] L-18. `Redis` client has no `socket_timeout`; shared unbounded pool for limiter + cache + arq + SSE
`redis_client.py:52-57` (only `socket_connect_timeout=3`). A hung Redis stalls every `await redis...` (rate limiter → every request; answer cache; SSE). Set `socket_timeout`, bound `max_connections`.

## [LOW] L-19. PreviewCache: process-local dict + eviction race + sync S3 delete on loop
`preview_cache.py:36` (in-memory registry — multi-instance would 404 previews created elsewhere), `:110-117` (`_evict` → sync `delete_file`), `:55-94` (check-TTL-then-download race with a concurrent `store()` cleanup → `NoSuchKey` surfaces as 500). Single-instance acceptable; move TTL state to Redis if scaling.

## [LOW] L-20. `BenchmarkSweepModel.best_run_id` has no FK
`models.py:321` (plain Integer; `k6l7m8n9o0p1:76`). Deleting a run leaves a dangling best_run_id.

## [LOW] L-21. Startup ordering: `ensure_collection` failure only warns; `container.dispose()` runs before `scheduler.shutdown()`
`main.py:99-107` (server starts with an unusable vector store — every search 500s until fixed, health stays green-ish), `main.py:129-130` (a 30-s tick firing between the two hits a disposed container; caught by `handle_exceptions`, but ordering is wrong).

## [LOW] L-22. Chat history context window: history saved only after a successful LLM stream
`chat_service.py:203-239` — if the client disconnects mid-stream the generator is cancelled → the turn (question + partial answer) is never persisted; subsequent condense/history lose that turn. Also two concurrent first-messages on a fresh conversation create two conversations (`conversation_repository.get_or_create:62-67`, no lock). Acceptable for chat UX, but document it.

---

# Security Findings (consolidated view)

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| S-1 | ACL OR-bug + `document_id` bypass in `search_substring` (API + chat exact boost) | CRITICAL | C-1 |
| S-2 | BM25 hash resolution without ACL (chat hybrid) | CRITICAL | C-2 |
| S-3 | Login brute force + X-Forwarded-For rate-limit bypass | HIGH | H-6 |
| S-4 | `/metrics` unauthenticated; label cardinality bomb | MEDIUM | M-4 |
| S-5 | Cross-user answer cache scope | MEDIUM | M-1 |
| S-6 | Raw PII in audit logs; SSE exception leak | MEDIUM | M-10 |
| S-7 | Existence oracles (404 vs 403/409) | LOW | L-4 |
| S-8 | Group delete/chunk-edit asymmetry | HIGH (policy) | H-12 |
| S-9 | Weak default credentials (compose, .env.example, admin/admin) | LOW | L-5, L-6 |
| S-10 | Unbounded upload buffering; no nginx body limit | MEDIUM | M-16, M-18 |
| S-11 | JWT/no revocation; `sub` parse 500 | LOW | L-1 |
| S-12 | Password policy / bcrypt truncation | LOW | L-2 |

Mass-assignment: none found (schemas reviewed; `owner_id` always server-derived; role/kind only on admin endpoints). Missing `require_admin`: none (all `/admin/*`, benchmark, ingest routes verified). API-key scoping: correct (revoke filtered by `user_id`, `api_key_repository.py:40-55`). Conversation/document/chunk object-level checks: correct outside S-1/S-2. SQL injection: clean — all runtime `text()` uses are parameterized (`chunk_repository.py:203-207`, outbox `:49-69`); `icontains` uses bound params (wildcard injection `%`/`_` only, minor). SSRF: no user-controllable URL fetches found. Deserialization: no pickle/yaml.load.

# DDD Findings

**Real, not cosmetic, but with decay points:**

1. **Layering is genuinely enforced** — domain is free of FastAPI/SQLAlchemy/Pydantic (verified: `domain/services/access_control.py`, entities are pure dataclasses; parser is a Protocol in `domain/services/document_parser.py` with the implementation in infrastructure). Application services depend on ports (`application/ports/*`); the UoW base lives in application, implemented in infrastructure.
2. **The canonical-ACL pattern is good design undermined by a drifted third copy** — `get_visibility_conditions` is the single source of truth, and the Qdrant filter and document SQL honor it, but the chunk SQL re-implements the AND/OR combination and got it wrong (C-1). This is the classic cost of translating the same contract by hand in three places. Consolidate into one translator per backend, both consuming `VisibilityCondition`, with a shared compile-level test.
3. **Anemic domain model** — `Document`/`User`/`Chunk` are data holders; behavior that belongs to the aggregate (visibility rules, access checks, status transitions) lives in module functions (`check_document_access` in `document_service.py:70-82`) and repositories. `Document.mark_done/mark_failed` exist but status transitions are also performed ad hoc via `update_status(...)` string params in repositories — dual truth for state transitions.
4. **Private-attribute reach-through**: `document_processor.py:179` — `self._domain_registry._profiles.get("general")._settings` — crosses the encapsulation boundary of the registry; add a public accessor.
5. **Generic god-UoW**: `UnitOfWork` carries all 17 repositories for every use case (`application/uow.py:44-59`) — convenient but makes transaction scope invisible (services "just use" everything, which is how H-4/H-5 happened).
6. **Business flow in presentation**: `helpers.upload_and_enqueue` (upload → job → enqueue) orchestrates a business use case in the presentation layer; `routes/ingest.py` and `routes/documents.py` inline the same pattern. Move behind an application service with one transaction boundary.
7. **Domain events exist but are barely used** — `uow.publish_event` is wired only for config changes; document lifecycle events would remove the manual BM25/outbox bookkeeping scattered in services.

# Database Findings

Positives: `users.email` UNIQUE; `api_keys` FK CASCADE; partial unique `ux_documents_active_slot` (with the M-11 caveat); outbox partial index exists in migration (but missing in ORM — L-14); `claim_batch` uses `FOR UPDATE SKIP LOCKED`; creation_date is `timestamptz` with server default `NOW()`.

Gaps already itemized: M-2 (config duplicates), M-11 (NULL-owner duplicates), L-10 (acts uniqueness), L-11 (chunk_index uniqueness), L-13 (RMW counter), L-14 (drift), L-15 (tz mix), L-20 (missing FK), M-6/M-7/M-8 (query shape), M-15 (pool/replica policy). Also: `chat_logs` ILIKE search has no trigram index (minor, admin-only); no `CHECK` that `documents.owner_id IS NOT NULL` ↔ visibility (free combination is enforced only in Python).

# Concurrency Findings

Itemized with A/B sequences: C-4 (job timeout), H-5 (nested tx → status regression), H-8 (double ingest), H-9 (outbox double-claim + lost update), H-10 (lock across LLM), H-12/M-11 (duplicate slots), M-12 (SSE), M-13 (cron duplication), M-14 (sweep double-submit + cosmetic cancel), L-13 (counter RMW), L-19 (preview eviction race), L-22 (concurrent conversation creation). Plus C-5 (cross-process BM25) as a memory-consistency issue.

# Performance Findings

- **Event-loop blocking** (biggest real-world impact): C-3, H-1, H-2, M-5, worker-side sync parse/OCR/S3-listing inside arq tasks (`ingestion_service.py:227, 241-248, 526-547`; `tasks.py:330-339` full-corpus `BM25Index(all_texts)` build on the loop) — with `max_jobs=4`, one scanned PDF freezes the other three jobs and arq's own health checks.
- **Unbounded queries/payloads**: M-6 (documents list), M-7 (conversation list aggregation), M-8 (duplicate detection), `get_all_contents()` full-table for BM25 rebuild (memory), rename outbox payload (M-9), sync chat/`stream` paths pass unbounded `question` to LLM (cost, not perf).
- **N+1**: M-6 reconcile + document list enrichment; `delete_internal_documents` per-row ORM delete (`document_repository.py:235-245`); `_unique_filename` loop (check-then-act loop, `document_service.py:488-499`).
- **Cardinality**: M-4 metrics label bomb.
- **Per-call client construction**: `qdrant_ops.py:156`, arq pool per enqueue (`queue.py:23`), TEI sync clients per call (`tei_clients.py:55, 89`), sweep progress pool per publish (`tasks.py:200-220`).

# Testing Gaps

**The suite is red right now**: `uv run pytest` → **10 failed, 638 passed** (verified). Failing:

```
tests/test_acl.py::TestCanViewDocument::test_unknown_visibility_raises        (DID NOT RAISE ValueError)
tests/test_container.py::TestSubscribeConfigEvents::test_subscribes_handlers_to_event_bus
tests/test_domain_entities.py::TestUserCreation::{4 tests}                    (AttributeError: 'User' object has no attribute 'can_be_created_by' — renamed to ensure_can_be_created_by)
tests/test_postgres_config_listener.py::{4 tests}                             (refetch/resync contract drift)
```

This means `lint.yml` (which runs `pytest -v` on push/PR) is failing on every run — the gate is being ignored, and any regression can slip through a red pipeline.

Structural gaps:
1. **No HTTP-layer tests** — zero `TestClient`/`httpx` ASGI tests; auth, status-code mapping, and routes are untested end-to-end.
2. **No real-database tests** — the ACL SQL bug (C-1) is invisible because `search_substring` is only exercised through fakes (`tests/fakes.py:133`); there is no repository-level test against a PG (even a transaction-wrapped sqlite/pg test would have caught the AND/OR error).
3. **No tests for**: rate limiting/brute force, API-key lifecycle, outbox stuck-recovery races, arq job cancellation, registry-vs-sync ordering (H-3), nested-transaction pipeline (H-5), cross-process BM25, pagination, PII redaction in logs, `/metrics` exposure.
4. **No concurrency tests** anywhere (project rules explicitly require them for concurrency-sensitive behavior).
5. Positive: strong domain-unit coverage (648 tests; ACL VO tests, outbox saga with fakes, RRF merge, splitters, parsers) — the unit layer is healthy; the seams are not.

# Technical Debt

- Three hand-written translations of the ACL contract (SQL doc, SQL chunk, Qdrant) — one already drifted (C-1); consolidate.
- `rag_service.py` (928 lines) and `benchmark_admin.py` (460 lines) are god modules; the RAG pipeline mixes retrieval, caching, policy, telemetry, and fallback logic in one class with `# noqa: C901`.
- Duplicated upload/job/enqueue orchestration between `helpers.py`, `routes/ingest.py`, `benchmark_admin.py`.
- Dual status-transition truth (entity methods vs stringly-typed `update_status`).
- Version/`VERSION` file read at import; `settings` singleton with a context-var override hack (`_settings_overrides`) used by sweeps — a global-mutable-state pattern that already caused H-11.
- Middleware stack: 3× `BaseHTTPMiddleware` + Instrumentator per request (task overhead); custom `MetricsMiddleware` partially duplicates Instrumentator.
- No alembic check that ORM metadata matches migrations (drift already present, L-14).

# Architecture Scores

| Area | Score | Rationale |
|------|-------|-----------|
| Architecture | 6/10 | Clean layering + DI container are real; cross-process coupling (BM25), presentation-layer orchestration, god modules, global event bus/singleton settings pull it down. |
| DDD | 5/10 | Ports/protocols and domain purity are genuine; anemic entities, hand-drifted ACL triplication, business flows in presentation. |
| Security | 3/10 | Two critical cross-tenant read paths, unthrottled login, unauthenticated metrics, weak defaults; good mass-assignment/injection hygiene can't offset broken ACL. |
| FastAPI | 6/10 | Thin routes, unified exception mapping, per-method UoW dependencies; blocked by blocking I/O in routes, SSE bugs, metrics middleware cardinality. |
| Database | 5/10 | Solid constraints overall and correct outbox claim SQL; NULL-distinct gaps, missing uniqueness for chunks/acts, tz mix, pool hardening, drift. |
| Transactions | 4/10 | UoW semantics are correct; real use cases violate boundaries (external calls in tx, nested tx, registry-before-sync, fire-and-forget enqueue). |
| Concurrency | 3/10 | Event loop freezing paths, job timeout holes, double-ingest, per-process state, no locks — the highest operational risk area. |
| Performance | 5/10 | Acceptable single-node behavior; unbounded queries, N+1s, in-memory BM25 build, label bomb. |
| Testing | 4/10 | 648 unit tests but 10 red, zero HTTP/DB integration, critical bug invisible, no concurrency/security tests. |
| Observability | 6/10 | Structured logs, request IDs, log buffer, Prometheus, config/outbox metrics; undermined by cardinality bomb, PII in logs, `/metrics` exposure. |
| Maintainability | 6/10 | Good module hygiene and docs; god services, duplicated ACL, stringly-typed status, red tests erode it. |
| Production Readiness | 4/10 | Single-process API with freeze paths, red CI, weak default creds, exposed ports, no multi-instance story. |

# Remediation Roadmap

**Immediate (days — highest risk):**
1. Fix C-1: `and_(*parts)` per condition; `document_id` as top-level AND; add compiled-SQL + behavioral ACL tests.
2. Fix C-2: apply `access_filter` in `_resolve_hash_to_doc`.
3. Fix H-11 (`rag.retriever_top_k` → `retriever_top_k`) — one-line, unblocks sweeps.
4. Make CI green: fix the 10 stale tests; make pytest a required check.
5. Rate-limit `/auth/login` (IP+email, Redis) and stop trusting `X-Forwarded-For` (strip to socket IP / set `forwarded_allow_ips`).
6. `asyncio.to_thread` around: OCR/analyze/render (C-3), Qdrant sync upserts (H-2), S3 delete/rename (H-1), scheduler metric calls (M-5).

**Short term (1–2 sprints):**
7. Explicit arq `job_timeout` per function + `CancelledError` handlers + heartbeat-based orphan recovery + `PROCESSING`-stuck reconciler (C-4).
8. Deterministic `_job_id` for all enqueues; pending-job re-enqueue safety net (H-7, H-8).
9. One transaction per use case: `_existing_uow` in `DocumentProcessor.process` (H-5); registry upsert moved into the sync transaction (H-3); per-document ingest transactions; `set_document_id_by_source` via outbox (H-4).
10. Storage mutations after commit with compensation (H-1); conditional outbox `recover_stuck` + raised stuck timeout (H-9).
11. Outbox DONE transition conditional on `status='indexing'`; rolling-summary LLM call outside the FOR UPDATE tx (H-10).
12. `/metrics` behind auth; switch `MetricsMiddleware` to route-template labels (M-4).
13. Answer-cache scope += `user_id` (M-1); fix SSE sweep timeout loop (M-12); sweep guard + cooperative cancel (M-14).

**Medium term:**
14. DB constraint batch: `NULLS NOT DISTINCT`/expression unique for documents and config_parameters, `UNIQUE (document_id, chunk_index)`, acts uniqueness + `is_current` partial unique, FK for `best_run_id`, tz-consistent columns (M-2, M-11, L-10, L-11, L-15, L-20).
15. BM25 invalidation channel (worker → API) or shared store (C-5); per-group BM25 scoping review.
16. Pagination everywhere (`GET /documents`), reconcile query rewrite, `find_duplicate_by_hash` by index (M-6, M-7, M-8).
17. Pool hardening (`pool_pre_ping`, `pool_recycle`, `statement_timeout`), replica read policy (M-15); Redis `socket_timeout` + bounded pools (L-18).
18. Integrate a real-PG test layer (transaction-rolled repository tests incl. compiled-ACL assertions) and a `TestClient` route test suite with authz matrix (Testing Gaps).
19. Compose hardening: rotate/remove default creds, bind MinIO console and API to loopback or private network, `client_max_body_size` (L-5, L-6, M-18).

**Long term (optional):**
20. Multi-instance readiness: advisory-lock cron jobs, single outbox-dispatch owner, Redis-backed preview cache (M-13, L-19).
21. Consolidate ACL translation into one codegen/tested adapter pair (SQL + Qdrant) consuming `VisibilityCondition` (DDD-2).
22. Enrich aggregates: move status transitions fully into entities; drop stringly-typed `update_status` overloads; use domain events for outbox/BM25 bookkeeping (DDD-3, DDD-7).
23. Split `rag_service.py` into retrieval / policy / cache / telemetry modules; extract benchmark orchestration from routes.

---

*Second pass completed: duplicates merged (S-3/H-6; db-agent findings #1/#2/#3/#4 folded into H-4/H-3/H-1/M-11), speculative items without code evidence dropped (e.g., "cache stampede" for per-id preview files, thread-tearing of BM25 mutation — downgraded/omitted), severity raised only where impact justified (BM25 staleness → CRITICAL due to silent search degradation in the default config; sweep Phase B → HIGH due to complete feature breakage).*
