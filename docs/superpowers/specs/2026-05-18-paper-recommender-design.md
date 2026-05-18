# Paper Recommender Design

## Summary

Build a low-cost paper recommendation web service using arXiv metadata. The service recommends papers similar to a user-provided arXiv URL by using metadata embeddings, not PDF full text. The target operating scale is about 100 daily users, so the design prioritizes zero-cost or very low-cost hosting while preserving recommendation quality through measured vector compression.

The recommended deployment is a free VM first, preferably Oracle Cloud Always Free. If that is not available or stable enough, the same file-based architecture should move to a low-cost VPS such as a small Hetzner instance.

## Goals

- Collect arXiv metadata through OAI-PMH.
- Embed only the latest `title + abstract + categories` for each paper.
- Keep one current vector per paper.
- Do not store historical vectors.
- Update daily using OAI-PMH `oai_datestamp`, not paper authored or published date.
- Re-embed modified papers only when `title`, `abstract`, or `categories` changes.
- Exclude deleted OAI-PMH records from recommendation results.
- Let users input an arXiv URL and receive top-k similar papers.
- Support optional category, date, and top-k filters.
- Use English UI labels and user-facing error messages.
- Prioritize monthly cost of `$0`, with a fallback target below `$10/month`.

## Non-Goals

- Do not collect or embed PDF full text.
- Do not store complete historical metadata snapshots.
- Do not store historical vectors.
- Do not build a high-traffic, horizontally scaled service in the first version.
- Do not require a managed vector database in the first version.
- Do not store full title and abstract for every paper solely for display.

## Operating Assumptions

- Expected traffic is about 100 daily users.
- Low QPS allows the search path to favor recall over latency.
- Initial full processing can run locally or on a separate machine.
- Initial backfill does not need to complete in a single run.
- Backfill can be split into OAI `datestamp` windows and resumed from checkpoints.
- arXiv metadata can be re-collected if needed, so the main backup priority is local state, model artifacts, and production indexes.

## Architecture

The system has four main parts:

1. OAI-PMH ingestion
2. Embedding and compression pipeline
3. SQLite metadata store plus local vector index files
4. Small web/API service

### OAI-PMH Ingestion

The ingestion pipeline fetches arXiv OAI-PMH records by `oai_datestamp`.

Initial backfill:

- Split the historical OAI `datestamp` range into smaller windows.
- Store checkpoint state for each window.
- Persist `resumptionToken` progress when present.
- Resume from the latest checkpoint after interruption.

Daily incremental update:

- Start from the last successful OAI `datestamp`.
- Fetch new, modified, and deleted records.
- Use OAI `datestamp` as the only incremental cursor.

Deleted records:

- Mark the paper inactive in SQLite.
- Add the vector id to a delete or tombstone table.
- Exclude the paper from recommendations immediately.
- Physically remove or compact deleted vectors during periodic index rebuilds if the selected index does not support efficient deletion.

### Embedding And Compression

For each active paper, the embedding input is:

```text
title
abstract
categories
```

The pipeline computes a `content_hash` from the normalized embedding input. Modified OAI records are re-embedded only when this hash changes.

The base embedding model is local and open source. The design should evaluate 384-dimensional and 768-dimensional model options before final selection.

The production vector index stores compressed vectors only:

1. Generate local embedding.
2. Apply PCA or OPQ dimensionality reduction.
3. Apply quantization, such as int8 scalar quantization or IVF-PQ.
4. Add or replace the paper vector in the staging index.

Auto-encoder compression is not part of the default design. It can be evaluated later only if it beats PCA/OPQ on recall while keeping operational complexity acceptable.

### Compression Quality Gate

Compression settings must be selected by measurement, not by storage size alone.

Create a balanced evaluation set across arXiv categories. Compare compressed search results against a float32 baseline.

Track at least:

- `recall@10`
- `recall@50`
- category-level recall variance
- self-neighbor sanity checks
- insufficient-result rate after filters

Because expected traffic is low, search parameters should favor recall:

- Use larger overfetch counts.
- Use larger `nprobe`, `ef_search`, or equivalent search breadth settings.
- Accept slower responses when the quality gain is meaningful.

A compressed index is promoted only after passing the recall gate.

## Storage Design

Use SQLite for the first version. It is simpler than Postgres and appropriate for the expected traffic level.

### `papers`

| Column | Purpose |
| --- | --- |
| `arxiv_id` | Primary key |
| `vector_id` | Local vector index id, nullable before embedding |
| `active` | Whether the paper can appear in recommendations |
| `oai_datestamp` | Latest OAI-PMH datestamp seen |
| `published_date` | Optional date filter field |
| `updated_date` | Optional date filter field |
| `primary_category` | Primary arXiv category |
| `categories` | Compact category list |
| `content_hash` | Hash of normalized title, abstract, and categories |
| `last_seen_at` | Local processing timestamp |

### `pipeline_state`

Stores operational cursors and version state.

Example keys:

- `last_successful_oai_datestamp`
- `current_backfill_window`
- `embedding_model_version`
- `compression_model_version`
- `production_index_version`

### `index_deletes`

Stores vector tombstones until the next compacting index rebuild.

| Column | Purpose |
| --- | --- |
| `arxiv_id` | Paper id |
| `vector_id` | Deleted vector id |
| `deleted_at` | Local deletion timestamp |

### `preview_cache`

Optional cache for recently viewed display metadata.

This table is not a full metadata store. It exists only to improve user experience for frequently viewed results.

| Column | Purpose |
| --- | --- |
| `arxiv_id` | Paper id |
| `title` | Cached title |
| `snippet` | Short cached snippet |
| `cached_at` | Cache timestamp |

The full title and abstract for every paper should not be stored for display in the first version. Recommendation result cards should rely primarily on arXiv links or page previews.

## Recommendation API

### Request Flow

1. Parse the submitted arXiv URL.
2. Normalize the `arxiv_id`.
3. Look up the paper in SQLite.
4. Reject missing, inactive, or vectorless records with a clear English error.
5. Retrieve the query vector from the local index.
6. Run ANN search with overfetch.
7. Exclude the query paper itself.
8. Exclude inactive or deleted papers.
9. Apply optional category and date filters.
10. Return up to `top_k` results.

### Supported URL Formats

- `https://arxiv.org/abs/1706.03762`
- `https://arxiv.org/pdf/1706.03762`
- `https://arxiv.org/abs/1706.03762v7`
- `https://arxiv.org/abs/cs/9901001`

Version suffixes are normalized away. For example, `1706.03762v7` becomes `1706.03762`.

### Result Fields

Return only compact operational fields:

- arXiv ID
- arXiv abs URL
- primary category
- categories
- published date
- updated date
- similarity score

The UI may render arXiv page previews or external links. Local storage does not need full display metadata for every paper.

## User Interface

The UI language is English.

Main labels:

- `arXiv URL`
- `Category`
- `Date range`
- `Top K`
- `Find similar papers`
- `Similar papers`
- `Open on arXiv`
- `No results`

User-facing errors:

- Invalid URL: `Please enter a valid arXiv URL, e.g. https://arxiv.org/abs/1706.03762`
- Unknown ID: `This arXiv ID is not available in the recommendation index yet. It may not have been backfilled or may have been removed.`
- Vector missing: `The recommendation vector for this paper is not ready yet. Please try again after the next index update.`
- Deleted record: `This paper is marked as deleted in arXiv OAI-PMH and is excluded from recommendations.`
- Not enough results: `Not enough similar papers match your filters. Try relaxing the category or date range.`

## Deployment

### Preferred: Oracle Cloud Always Free

Target cost: `$0/month`.

Use one Always Free VM if available. The free ARM allocation is suitable for a low-traffic service and a compressed local index. Keep SQLite, production index files, logs, and recent backups within the free block volume allowance.

Risks:

- Always Free capacity may be unavailable.
- Idle resources may be reclaimed.
- Operational reliability is lower than a paid VPS.

Mitigation:

- Keep the deployment file-based and portable.
- Back up SQLite, index artifacts, and model artifacts.
- Use lightweight health checks.

### Fallback: Low-Cost VPS

Target cost: below `$10/month`.

A small Hetzner instance is the preferred fallback. The service should not depend on managed database or managed vector infrastructure, so migration should mostly involve copying files and restarting the service.

## Backup And Rollback

Back up:

- SQLite database
- production vector index files
- PCA/OPQ artifacts
- quantization parameters
- embedding model version/config
- index build config

Do not treat full OAI metadata as critical backup data because it can be re-collected.

Use versioned production indexes:

1. Build a staging index.
2. Run quality checks.
3. Swap staging to production atomically.
4. Keep at least one previous production version for rollback.

## Testing Strategy

### Unit Tests

URL parsing:

- abs URLs
- pdf URLs
- old-style IDs
- version suffixes
- invalid URLs

OAI parsing:

- normal records
- deleted records
- `resumptionToken`
- datestamp window boundaries

Update logic:

- new paper
- unchanged modified record
- changed modified record
- deleted paper

Recommendation logic:

- self exclusion
- inactive exclusion
- category filter
- date filter
- insufficient results
- missing vector

### Quality Tests

Compression quality:

- float32 baseline versus compressed index
- `recall@10`
- `recall@50`
- per-category recall
- filter result availability

Promotion rule:

- Do not promote a compressed index unless it passes the configured recall threshold.
- If recall is too low, increase retained dimensions, relax quantization, or increase search breadth.

## Open Implementation Choices

These should be decided during implementation planning:

- Exact local embedding model.
- Exact vector library: FAISS or USearch.
- Exact compression configuration: PCA dimension, OPQ usage, int8 versus IVF-PQ.
- Exact recall thresholds for promotion.
- Exact deployment target after checking Oracle Always Free availability.

## Success Criteria

- The service runs at `$0/month` if Oracle Always Free is available.
- Fallback monthly cost remains below `$10/month`.
- The system supports about 100 daily users.
- OAI-PMH incremental updates use `oai_datestamp`.
- Each paper has at most one latest active vector.
- Deleted OAI records are excluded from recommendations.
- Modified papers are re-embedded only when `title + abstract + categories` changes.
- Recommendation API excludes the input paper itself.
- Missing or unavailable IDs return clear English errors.
- Compressed index quality is measured before production promotion.
