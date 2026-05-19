# Paper Recommender MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable MVP that accepts an arXiv URL, finds the matching local vector, and returns filtered top-k similar papers from a local SQLite database and exact local vector index.

**Architecture:** Use a small Python FastAPI app with focused modules for URL parsing, metadata storage, OAI record parsing, vector search, and recommendation orchestration. The MVP uses exact cosine search over a local `.npz` index so API behavior can be tested before production OAI backfill, PCA/OPQ compression, and quantized ANN indexing are added in separate plans.

**Tech Stack:** Python 3.11+, FastAPI, SQLite, NumPy, pytest, FastAPI TestClient, static HTML/CSS/JavaScript.

---

## Scope Check

This plan implements the first working product slice:

- Project scaffold and test harness
- arXiv URL parsing
- SQLite schema and paper state operations
- OAI-PMH XML parsing and content-hash update decisions
- Exact local vector index abstraction
- Recommendation service rules
- FastAPI recommendation endpoint
- Minimal English UI
- Sample seed data for local manual testing

This plan intentionally excludes production-scale OAI backfill, PCA/OPQ training, quantized ANN indexing, deployment automation, and backup automation. Those are separate subsystems with independent quality gates. The MVP creates the interfaces those follow-up plans will replace or extend.

## File Structure

Create this structure:

```text
pyproject.toml
src/paper_recommender/__init__.py
src/paper_recommender/arxiv_id.py
src/paper_recommender/app.py
src/paper_recommender/models.py
src/paper_recommender/oai.py
src/paper_recommender/pipeline.py
src/paper_recommender/recommender.py
src/paper_recommender/storage.py
src/paper_recommender/vector_store.py
src/paper_recommender/static/index.html
src/paper_recommender/static/app.js
src/paper_recommender/static/styles.css
scripts/seed_sample_data.py
tests/fixtures/oai_records.xml
tests/test_arxiv_id.py
tests/test_storage.py
tests/test_oai.py
tests/test_pipeline.py
tests/test_vector_store.py
tests/test_recommender.py
tests/test_app.py
tests/test_static_ui.py
```

Responsibilities:

- `arxiv_id.py`: Parse and normalize supported arXiv URLs.
- `models.py`: Shared dataclasses and constants.
- `storage.py`: SQLite schema and persistence functions.
- `oai.py`: Parse OAI-PMH XML into normalized records.
- `pipeline.py`: Compute content hashes and apply OAI record changes to SQLite.
- `vector_store.py`: Exact cosine vector index with `.npz` save/load.
- `recommender.py`: Business rules for recommendation and filtering.
- `app.py`: FastAPI application factory and API routes.
- `static/*`: Minimal English UI.
- `scripts/seed_sample_data.py`: Local sample dataset and vector index for manual testing.

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/paper_recommender/__init__.py`

- [ ] **Step 1: Add Python project metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "paper-recommender"
version = "0.1.0"
description = "Low-cost paper recommendation MVP using arXiv metadata."
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115,<1",
  "numpy>=1.26,<3",
  "uvicorn[standard]>=0.30,<1"
]

[project.optional-dependencies]
dev = [
  "httpx>=0.27,<1",
  "pytest>=8,<9",
  "ruff>=0.8,<1"
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Add package marker**

Create `src/paper_recommender/__init__.py`:

```python
"""Paper Recommender MVP package."""

__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 3: Install development dependencies**

Run:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Expected: packages install without errors.

- [ ] **Step 4: Run empty test suite**

Run:

```powershell
.\.venv\Scripts\python -m pytest
```

Expected: pytest starts and reports no tests collected or no failures.

- [ ] **Step 5: Commit scaffold**

```powershell
git add pyproject.toml src/paper_recommender/__init__.py
git commit -m "Add Python project scaffold"
```

---

### Task 2: arXiv URL Parser

**Files:**
- Create: `src/paper_recommender/arxiv_id.py`
- Create: `tests/test_arxiv_id.py`

- [ ] **Step 1: Write parser tests**

Create `tests/test_arxiv_id.py`:

```python
import pytest

from paper_recommender.arxiv_id import InvalidArxivUrl, parse_arxiv_id


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762v7", "1706.03762"),
        ("https://www.arxiv.org/abs/cs/9901001", "cs/9901001"),
        ("https://arxiv.org/pdf/hep-th/9901001.pdf", "hep-th/9901001"),
    ],
)
def test_parse_supported_arxiv_urls(url: str, expected: str) -> None:
    assert parse_arxiv_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "1706.03762",
        "https://example.com/abs/1706.03762",
        "https://arxiv.org/list/cs.AI/recent",
        "https://arxiv.org/abs/not-an-id",
    ],
)
def test_reject_invalid_urls(url: str) -> None:
    with pytest.raises(InvalidArxivUrl) as exc_info:
        parse_arxiv_id(url)

    assert "Please enter a valid arXiv URL" in str(exc_info.value)
```

- [ ] **Step 2: Run parser tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_arxiv_id.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing `parse_arxiv_id`.

- [ ] **Step 3: Implement parser**

Create `src/paper_recommender/arxiv_id.py`:

```python
from __future__ import annotations

import re
from urllib.parse import urlparse

INVALID_ARXIV_URL_MESSAGE = (
    "Please enter a valid arXiv URL, e.g. https://arxiv.org/abs/1706.03762"
)

_MODERN_ID_RE = re.compile(r"^(?P<id>\d{4}\.\d{4,5})(?:v\d+)?$")
_OLD_ID_RE = re.compile(r"^(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


class InvalidArxivUrl(ValueError):
    """Raised when a URL does not point to a supported arXiv abs or pdf path."""


def parse_arxiv_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)
    if parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org"}:
        raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)

    path = parsed.path.strip("/")
    if path.startswith("abs/"):
        raw_id = path.removeprefix("abs/")
    elif path.startswith("pdf/"):
        raw_id = path.removeprefix("pdf/")
        raw_id = raw_id.removesuffix(".pdf")
    else:
        raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)

    for pattern in (_MODERN_ID_RE, _OLD_ID_RE):
        match = pattern.match(raw_id)
        if match:
            return match.group("id")

    raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)
```

- [ ] **Step 4: Run parser tests to verify pass**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_arxiv_id.py -v
```

Expected: all parser tests PASS.

- [ ] **Step 5: Commit parser**

```powershell
git add src/paper_recommender/arxiv_id.py tests/test_arxiv_id.py
git commit -m "Add arXiv URL parser"
```

---

### Task 3: SQLite Storage

**Files:**
- Create: `src/paper_recommender/models.py`
- Create: `src/paper_recommender/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write storage tests**

Create `tests/test_storage.py`:

```python
from paper_recommender.models import Paper
from paper_recommender.storage import (
    connect_db,
    get_paper,
    get_paper_by_vector_id,
    init_db,
    mark_deleted,
    update_oai_datestamp,
    upsert_paper,
)


def test_upsert_and_get_paper() -> None:
    conn = connect_db(":memory:")
    init_db(conn)

    paper = Paper(
        arxiv_id="1706.03762",
        vector_id=1,
        active=True,
        oai_datestamp="2024-01-02",
        published_date="2017-06-12",
        updated_date="2023-08-02",
        primary_category="cs.CL",
        categories=("cs.CL", "cs.LG"),
        content_hash="hash-a",
    )
    upsert_paper(conn, paper)

    stored = get_paper(conn, "1706.03762")

    assert stored == paper
    assert get_paper_by_vector_id(conn, 1) == paper


def test_mark_deleted_deactivates_paper_and_records_tombstone() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-a",
        ),
    )

    mark_deleted(conn, "1706.03762", "2024-01-03")

    stored = get_paper(conn, "1706.03762")
    tombstones = conn.execute("SELECT arxiv_id, vector_id FROM index_deletes").fetchall()

    assert stored is not None
    assert stored.active is False
    assert tombstones == [("1706.03762", 1)]


def test_update_oai_datestamp_without_reembedding() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-a",
        ),
    )

    update_oai_datestamp(conn, "1706.03762", "2024-01-04")

    stored = get_paper(conn, "1706.03762")
    assert stored is not None
    assert stored.oai_datestamp == "2024-01-04"
    assert stored.content_hash == "hash-a"
```

- [ ] **Step 2: Run storage tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_storage.py -v
```

Expected: FAIL with missing `models` or `storage` module.

- [ ] **Step 3: Add shared models**

Create `src/paper_recommender/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_ID_MESSAGE = (
    "This arXiv ID is not available in the recommendation index yet. "
    "It may not have been backfilled or may have been removed."
)
VECTOR_MISSING_MESSAGE = (
    "The recommendation vector for this paper is not ready yet. "
    "Please try again after the next index update."
)
DELETED_RECORD_MESSAGE = (
    "This paper is marked as deleted in arXiv OAI-PMH and is excluded from recommendations."
)
NOT_ENOUGH_RESULTS_MESSAGE = (
    "Not enough similar papers match your filters. Try relaxing the category or date range."
)


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    vector_id: int | None
    active: bool
    oai_datestamp: str
    published_date: str | None
    updated_date: str | None
    primary_category: str
    categories: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True)
class Recommendation:
    arxiv_id: str
    url: str
    primary_category: str
    categories: tuple[str, ...]
    published_date: str | None
    updated_date: str | None
    similarity_score: float
```

- [ ] **Step 4: Implement SQLite storage**

Create `src/paper_recommender/storage.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from paper_recommender.models import Paper


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    vector_id INTEGER UNIQUE,
    active INTEGER NOT NULL,
    oai_datestamp TEXT NOT NULL,
    published_date TEXT,
    updated_date TEXT,
    primary_category TEXT NOT NULL,
    categories TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_deletes (
    arxiv_id TEXT NOT NULL,
    vector_id INTEGER,
    deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS preview_cache (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    snippet TEXT NOT NULL,
    cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _encode_categories(categories: tuple[str, ...]) -> str:
    return " ".join(categories)


def _decode_categories(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(" ") if part)


def _row_to_paper(row: sqlite3.Row | None) -> Paper | None:
    if row is None:
        return None
    return Paper(
        arxiv_id=row["arxiv_id"],
        vector_id=row["vector_id"],
        active=bool(row["active"]),
        oai_datestamp=row["oai_datestamp"],
        published_date=row["published_date"],
        updated_date=row["updated_date"],
        primary_category=row["primary_category"],
        categories=_decode_categories(row["categories"]),
        content_hash=row["content_hash"],
    )


def upsert_paper(conn: sqlite3.Connection, paper: Paper) -> None:
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, vector_id, active, oai_datestamp, published_date, updated_date,
            primary_category, categories, content_hash, last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            vector_id = excluded.vector_id,
            active = excluded.active,
            oai_datestamp = excluded.oai_datestamp,
            published_date = excluded.published_date,
            updated_date = excluded.updated_date,
            primary_category = excluded.primary_category,
            categories = excluded.categories,
            content_hash = excluded.content_hash,
            last_seen_at = CURRENT_TIMESTAMP
        """,
        (
            paper.arxiv_id,
            paper.vector_id,
            int(paper.active),
            paper.oai_datestamp,
            paper.published_date,
            paper.updated_date,
            paper.primary_category,
            _encode_categories(paper.categories),
            paper.content_hash,
        ),
    )
    conn.commit()


def get_paper(conn: sqlite3.Connection, arxiv_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return _row_to_paper(row)


def get_paper_by_vector_id(conn: sqlite3.Connection, vector_id: int) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE vector_id = ?", (vector_id,)).fetchone()
    return _row_to_paper(row)


def update_oai_datestamp(conn: sqlite3.Connection, arxiv_id: str, oai_datestamp: str) -> None:
    conn.execute(
        """
        UPDATE papers
        SET oai_datestamp = ?, last_seen_at = CURRENT_TIMESTAMP
        WHERE arxiv_id = ?
        """,
        (oai_datestamp, arxiv_id),
    )
    conn.commit()


def mark_deleted(conn: sqlite3.Connection, arxiv_id: str, oai_datestamp: str) -> None:
    existing = get_paper(conn, arxiv_id)
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, vector_id, active, oai_datestamp, published_date, updated_date,
            primary_category, categories, content_hash, last_seen_at
        )
        VALUES (?, NULL, 0, ?, NULL, NULL, '', '', '', CURRENT_TIMESTAMP)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            active = 0,
            oai_datestamp = excluded.oai_datestamp,
            last_seen_at = CURRENT_TIMESTAMP
        """,
        (arxiv_id, oai_datestamp),
    )
    if existing and existing.vector_id is not None:
        conn.execute(
            "INSERT INTO index_deletes (arxiv_id, vector_id) VALUES (?, ?)",
            (arxiv_id, existing.vector_id),
        )
    conn.commit()


def set_pipeline_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_state (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


def get_pipeline_state(conn: sqlite3.Connection, key: str) -> str | None:
    row: Any = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]
```

- [ ] **Step 5: Run storage tests to verify pass**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_storage.py -v
```

Expected: all storage tests PASS.

- [ ] **Step 6: Commit storage**

```powershell
git add src/paper_recommender/models.py src/paper_recommender/storage.py tests/test_storage.py
git commit -m "Add SQLite paper storage"
```

---

### Task 4: OAI Parsing And Update Decisions

**Files:**
- Create: `src/paper_recommender/oai.py`
- Create: `src/paper_recommender/pipeline.py`
- Create: `tests/fixtures/oai_records.xml`
- Create: `tests/test_oai.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Add OAI XML fixture**

Create `tests/fixtures/oai_records.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:1706.03762</identifier>
        <datestamp>2024-01-02</datestamp>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>1706.03762</id>
          <created>2017-06-12</created>
          <updated>2023-08-02</updated>
          <title>Attention Is All You Need</title>
          <abstract>We propose a new simple network architecture.</abstract>
          <categories>cs.CL cs.LG</categories>
        </arXiv>
      </metadata>
    </record>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:9999.00001</identifier>
        <datestamp>2024-01-03</datestamp>
      </header>
    </record>
    <resumptionToken cursor="0" completeListSize="2">abc123</resumptionToken>
  </ListRecords>
</OAI-PMH>
```

- [ ] **Step 2: Write OAI parser tests**

Create `tests/test_oai.py`:

```python
from pathlib import Path

from paper_recommender.oai import parse_oai_records


def test_parse_normal_and_deleted_records() -> None:
    xml = Path("tests/fixtures/oai_records.xml").read_text(encoding="utf-8")

    batch = parse_oai_records(xml)

    assert batch.resumption_token == "abc123"
    assert len(batch.records) == 2
    assert batch.records[0].arxiv_id == "1706.03762"
    assert batch.records[0].deleted is False
    assert batch.records[0].title == "Attention Is All You Need"
    assert batch.records[0].categories == ("cs.CL", "cs.LG")
    assert batch.records[1].arxiv_id == "9999.00001"
    assert batch.records[1].deleted is True
```

- [ ] **Step 3: Write pipeline tests**

Create `tests/test_pipeline.py`:

```python
from paper_recommender.models import Paper
from paper_recommender.oai import OaiRecord
from paper_recommender.pipeline import apply_oai_record, compute_content_hash
from paper_recommender.storage import connect_db, get_paper, init_db, upsert_paper


def test_content_hash_ignores_whitespace_noise() -> None:
    first = compute_content_hash("A  Title", "Line one\nline two", ("cs.CL", "cs.LG"))
    second = compute_content_hash("A Title", "Line one line two", ("cs.LG", "cs.CL"))

    assert first == second


def test_apply_new_record_inserts_paper_without_vector() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    record = OaiRecord(
        arxiv_id="1706.03762",
        oai_datestamp="2024-01-02",
        deleted=False,
        title="Attention",
        abstract="Abstract",
        categories=("cs.CL",),
        published_date="2017-06-12",
        updated_date="2023-08-02",
    )

    decision = apply_oai_record(conn, record)

    stored = get_paper(conn, "1706.03762")
    assert decision == "inserted"
    assert stored is not None
    assert stored.vector_id is None
    assert stored.active is True


def test_apply_unchanged_record_updates_datestamp_only() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    content_hash = compute_content_hash("Attention", "Abstract", ("cs.CL",))
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date="2023-08-02",
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash=content_hash,
        ),
    )

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="1706.03762",
            oai_datestamp="2024-01-04",
            deleted=False,
            title="Attention",
            abstract="Abstract",
            categories=("cs.CL",),
            published_date="2017-06-12",
            updated_date="2023-08-02",
        ),
    )

    stored = get_paper(conn, "1706.03762")
    assert decision == "unchanged"
    assert stored is not None
    assert stored.vector_id == 1
    assert stored.oai_datestamp == "2024-01-04"


def test_apply_changed_record_clears_vector_for_reembedding() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="old-hash",
        ),
    )

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="1706.03762",
            oai_datestamp="2024-01-05",
            deleted=False,
            title="New title",
            abstract="New abstract",
            categories=("cs.CL", "cs.LG"),
            published_date=None,
            updated_date=None,
        ),
    )

    stored = get_paper(conn, "1706.03762")
    assert decision == "updated"
    assert stored is not None
    assert stored.vector_id is None
    assert stored.categories == ("cs.CL", "cs.LG")


def test_apply_deleted_record_marks_inactive() -> None:
    conn = connect_db(":memory:")
    init_db(conn)

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="9999.00001",
            oai_datestamp="2024-01-03",
            deleted=True,
            title=None,
            abstract=None,
            categories=(),
            published_date=None,
            updated_date=None,
        ),
    )

    stored = get_paper(conn, "9999.00001")
    assert decision == "deleted"
    assert stored is not None
    assert stored.active is False
```

- [ ] **Step 4: Run OAI and pipeline tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_oai.py tests/test_pipeline.py -v
```

Expected: FAIL with missing `oai` or `pipeline` modules.

- [ ] **Step 5: Implement OAI parser**

Create `src/paper_recommender/oai.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET


OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}


@dataclass(frozen=True)
class OaiRecord:
    arxiv_id: str
    oai_datestamp: str
    deleted: bool
    title: str | None
    abstract: str | None
    categories: tuple[str, ...]
    published_date: str | None
    updated_date: str | None


@dataclass(frozen=True)
class OaiBatch:
    records: tuple[OaiRecord, ...]
    resumption_token: str | None


def _text(parent: ET.Element, path: str) -> str | None:
    node = parent.find(path, OAI_NS)
    if node is None or node.text is None:
        return None
    return " ".join(node.text.split())


def _arxiv_id_from_header(identifier: str) -> str:
    return identifier.removeprefix("oai:arXiv.org:")


def parse_oai_records(xml: str) -> OaiBatch:
    root = ET.fromstring(xml)
    records: list[OaiRecord] = []
    for record in root.findall(".//oai:record", OAI_NS):
        header = record.find("oai:header", OAI_NS)
        if header is None:
            continue
        identifier = _text(header, "oai:identifier")
        datestamp = _text(header, "oai:datestamp")
        if identifier is None or datestamp is None:
            continue
        arxiv_id = _arxiv_id_from_header(identifier)
        deleted = header.attrib.get("status") == "deleted"
        if deleted:
            records.append(
                OaiRecord(
                    arxiv_id=arxiv_id,
                    oai_datestamp=datestamp,
                    deleted=True,
                    title=None,
                    abstract=None,
                    categories=(),
                    published_date=None,
                    updated_date=None,
                )
            )
            continue

        metadata = record.find("oai:metadata/ax:arXiv", OAI_NS)
        if metadata is None:
            continue
        categories_text = _text(metadata, "ax:categories") or ""
        records.append(
            OaiRecord(
                arxiv_id=_text(metadata, "ax:id") or arxiv_id,
                oai_datestamp=datestamp,
                deleted=False,
                title=_text(metadata, "ax:title") or "",
                abstract=_text(metadata, "ax:abstract") or "",
                categories=tuple(part for part in categories_text.split(" ") if part),
                published_date=_text(metadata, "ax:created"),
                updated_date=_text(metadata, "ax:updated"),
            )
        )

    token = _text(root, ".//oai:resumptionToken")
    return OaiBatch(records=tuple(records), resumption_token=token)
```

- [ ] **Step 6: Implement pipeline decisions**

Create `src/paper_recommender/pipeline.py`:

```python
from __future__ import annotations

import hashlib
import sqlite3

from paper_recommender.models import Paper
from paper_recommender.oai import OaiRecord
from paper_recommender.storage import get_paper, mark_deleted, update_oai_datestamp, upsert_paper


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def compute_content_hash(title: str, abstract: str, categories: tuple[str, ...]) -> str:
    payload = "\n".join(
        [
            _normalize_text(title),
            _normalize_text(abstract),
            " ".join(sorted(categories)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_oai_record(conn: sqlite3.Connection, record: OaiRecord) -> str:
    if record.deleted:
        mark_deleted(conn, record.arxiv_id, record.oai_datestamp)
        return "deleted"

    title = record.title or ""
    abstract = record.abstract or ""
    content_hash = compute_content_hash(title, abstract, record.categories)
    existing = get_paper(conn, record.arxiv_id)
    if existing and existing.content_hash == content_hash:
        update_oai_datestamp(conn, record.arxiv_id, record.oai_datestamp)
        return "unchanged"

    paper = Paper(
        arxiv_id=record.arxiv_id,
        vector_id=None,
        active=True,
        oai_datestamp=record.oai_datestamp,
        published_date=record.published_date,
        updated_date=record.updated_date,
        primary_category=record.categories[0] if record.categories else "",
        categories=record.categories,
        content_hash=content_hash,
    )
    upsert_paper(conn, paper)
    return "inserted" if existing is None else "updated"
```

- [ ] **Step 7: Run OAI and pipeline tests to verify pass**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_oai.py tests/test_pipeline.py -v
```

Expected: all OAI and pipeline tests PASS.

- [ ] **Step 8: Commit OAI parsing and update decisions**

```powershell
git add src/paper_recommender/oai.py src/paper_recommender/pipeline.py tests/fixtures/oai_records.xml tests/test_oai.py tests/test_pipeline.py
git commit -m "Add OAI record parsing and update decisions"
```

---

### Task 5: Exact Local Vector Index

**Files:**
- Create: `src/paper_recommender/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write vector store tests**

Create `tests/test_vector_store.py`:

```python
import numpy as np

from paper_recommender.vector_store import ExactVectorIndex


def test_exact_vector_search_orders_by_cosine_similarity() -> None:
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.8, 0.2], dtype=np.float32),
            3: np.array([0.0, 1.0], dtype=np.float32),
        }
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [item.vector_id for item in results] == [1, 2]
    assert results[0].score > results[1].score


def test_vector_index_save_and_load(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    index = ExactVectorIndex.from_items({7: np.array([3.0, 4.0], dtype=np.float32)})

    index.save(path)
    loaded = ExactVectorIndex.load(path)

    assert np.allclose(loaded.get(7), np.array([0.6, 0.8], dtype=np.float32))
```

- [ ] **Step 2: Run vector tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_vector_store.py -v
```

Expected: FAIL with missing `vector_store`.

- [ ] **Step 3: Implement exact vector index**

Create `src/paper_recommender/vector_store.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    vector_id: int
    score: float


class ExactVectorIndex:
    def __init__(self, vector_ids: np.ndarray, vectors: np.ndarray) -> None:
        self._vector_ids = vector_ids.astype(np.int64)
        self._vectors = _normalize_matrix(vectors.astype(np.float32))
        self._positions = {int(vector_id): idx for idx, vector_id in enumerate(self._vector_ids)}

    @classmethod
    def from_items(cls, items: dict[int, np.ndarray]) -> "ExactVectorIndex":
        vector_ids = np.array(list(items.keys()), dtype=np.int64)
        vectors = np.vstack([items[int(vector_id)] for vector_id in vector_ids]).astype(np.float32)
        return cls(vector_ids, vectors)

    @classmethod
    def load(cls, path: str | Path) -> "ExactVectorIndex":
        data = np.load(path)
        return cls(data["vector_ids"], data["vectors"])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, vector_ids=self._vector_ids, vectors=self._vectors)

    def get(self, vector_id: int) -> np.ndarray | None:
        position = self._positions.get(vector_id)
        if position is None:
            return None
        return self._vectors[position].copy()

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        if top_k <= 0:
            return []
        normalized_query = _normalize_vector(query.astype(np.float32))
        scores = self._vectors @ normalized_query
        count = min(top_k, len(scores))
        if count == 0:
            return []
        top_positions = np.argpartition(-scores, count - 1)[:count]
        ordered_positions = top_positions[np.argsort(-scores[top_positions])]
        return [
            VectorSearchResult(
                vector_id=int(self._vector_ids[position]),
                score=float(scores[position]),
            )
            for position in ordered_positions
        ]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
```

- [ ] **Step 4: Run vector tests to verify pass**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_vector_store.py -v
```

Expected: all vector tests PASS.

- [ ] **Step 5: Commit vector index**

```powershell
git add src/paper_recommender/vector_store.py tests/test_vector_store.py
git commit -m "Add exact local vector index"
```

---

### Task 6: Recommendation Service

**Files:**
- Create: `src/paper_recommender/recommender.py`
- Create: `tests/test_recommender.py`

- [ ] **Step 1: Write recommendation tests**

Create `tests/test_recommender.py`:

```python
import numpy as np
import pytest

from paper_recommender.models import Paper, UNKNOWN_ID_MESSAGE, VECTOR_MISSING_MESSAGE
from paper_recommender.recommender import RecommendationError, recommend
from paper_recommender.storage import connect_db, init_db, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex


def _paper(arxiv_id: str, vector_id: int | None, category: str, date: str = "2024-01-01") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        vector_id=vector_id,
        active=True,
        oai_datestamp=date,
        published_date=date,
        updated_date=date,
        primary_category=category,
        categories=(category,),
        content_hash=f"hash-{arxiv_id}",
    )


def test_recommend_excludes_query_paper_and_applies_top_k() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    for paper in [
        _paper("1706.03762", 1, "cs.CL"),
        _paper("1111.11111", 2, "cs.CL"),
        _paper("2222.22222", 3, "cs.LG"),
    ]:
        upsert_paper(conn, paper)
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
            3: np.array([0.0, 1.0], dtype=np.float32),
        }
    )

    results = recommend(conn, index, "1706.03762", top_k=1)

    assert [result.arxiv_id for result in results] == ["1111.11111"]


def test_recommend_applies_category_filter() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    for paper in [
        _paper("1706.03762", 1, "cs.CL"),
        _paper("1111.11111", 2, "cs.CL"),
        _paper("2222.22222", 3, "cs.LG"),
    ]:
        upsert_paper(conn, paper)
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
            3: np.array([0.8, 0.2], dtype=np.float32),
        }
    )

    results = recommend(conn, index, "1706.03762", top_k=5, category="cs.LG")

    assert [result.arxiv_id for result in results] == ["2222.22222"]


def test_recommend_rejects_missing_paper() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    index = ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)})

    with pytest.raises(RecommendationError) as exc_info:
        recommend(conn, index, "missing", top_k=10)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == UNKNOWN_ID_MESSAGE


def test_recommend_rejects_vectorless_paper() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(conn, _paper("1706.03762", None, "cs.CL"))
    index = ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)})

    with pytest.raises(RecommendationError) as exc_info:
        recommend(conn, index, "1706.03762", top_k=10)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == VECTOR_MISSING_MESSAGE
```

- [ ] **Step 2: Run recommendation tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_recommender.py -v
```

Expected: FAIL with missing `recommender`.

- [ ] **Step 3: Implement recommendation service**

Create `src/paper_recommender/recommender.py`:

```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from paper_recommender.models import (
    DELETED_RECORD_MESSAGE,
    UNKNOWN_ID_MESSAGE,
    VECTOR_MISSING_MESSAGE,
    Recommendation,
)
from paper_recommender.storage import get_paper, get_paper_by_vector_id
from paper_recommender.vector_store import ExactVectorIndex


@dataclass(frozen=True)
class RecommendationError(Exception):
    status_code: int
    message: str


def recommend(
    conn: sqlite3.Connection,
    index: ExactVectorIndex,
    arxiv_id: str,
    top_k: int,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Recommendation]:
    query_paper = get_paper(conn, arxiv_id)
    if query_paper is None:
        raise RecommendationError(404, UNKNOWN_ID_MESSAGE)
    if not query_paper.active:
        raise RecommendationError(404, DELETED_RECORD_MESSAGE)
    if query_paper.vector_id is None:
        raise RecommendationError(404, VECTOR_MISSING_MESSAGE)

    query_vector = index.get(query_paper.vector_id)
    if query_vector is None:
        raise RecommendationError(404, VECTOR_MISSING_MESSAGE)

    overfetch = max(top_k * 20, 100)
    results: list[Recommendation] = []
    for vector_result in index.search(query_vector, top_k=overfetch):
        candidate = get_paper_by_vector_id(conn, vector_result.vector_id)
        if candidate is None:
            continue
        if candidate.arxiv_id == arxiv_id:
            continue
        if not candidate.active:
            continue
        if category and category not in candidate.categories:
            continue
        if date_from and candidate.published_date and candidate.published_date < date_from:
            continue
        if date_to and candidate.published_date and candidate.published_date > date_to:
            continue
        results.append(
            Recommendation(
                arxiv_id=candidate.arxiv_id,
                url=f"https://arxiv.org/abs/{candidate.arxiv_id}",
                primary_category=candidate.primary_category,
                categories=candidate.categories,
                published_date=candidate.published_date,
                updated_date=candidate.updated_date,
                similarity_score=vector_result.score,
            )
        )
        if len(results) >= top_k:
            break
    return results
```

- [ ] **Step 4: Run recommendation tests to verify pass**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_recommender.py -v
```

Expected: all recommendation tests PASS.

- [ ] **Step 5: Commit recommendation service**

```powershell
git add src/paper_recommender/recommender.py tests/test_recommender.py
git commit -m "Add recommendation service"
```

---

### Task 7: FastAPI Recommendation API

**Files:**
- Create: `src/paper_recommender/app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write API tests**

Create `tests/test_app.py`:

```python
import numpy as np
from fastapi.testclient import TestClient

from paper_recommender.app import create_app
from paper_recommender.models import Paper
from paper_recommender.storage import connect_db, init_db, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex


def _build_client(tmp_path) -> TestClient:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date="2023-08-02",
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-1",
        ),
    )
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1111.11111",
            vector_id=2,
            active=True,
            oai_datestamp="2024-01-03",
            published_date="2020-01-01",
            updated_date="2020-01-01",
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-2",
        ),
    )
    conn.close()
    ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    ).save(index_path)
    return TestClient(create_app(db_path=db_path, index_path=index_path))


def test_health_endpoint(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_results(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["query_arxiv_id"] == "1706.03762"
    assert response.json()["results"][0]["arxiv_id"] == "1111.11111"


def test_recommend_endpoint_rejects_invalid_url(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post("/api/recommend", json={"url": "https://example.com/nope"})

    assert response.status_code == 400
    assert "Please enter a valid arXiv URL" in response.json()["detail"]
```

- [ ] **Step 2: Run API tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_app.py -v
```

Expected: FAIL with missing `app`.

- [ ] **Step 3: Implement FastAPI app**

Create `src/paper_recommender/app.py`:

```python
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from paper_recommender.arxiv_id import InvalidArxivUrl, parse_arxiv_id
from paper_recommender.recommender import RecommendationError, recommend
from paper_recommender.storage import connect_db
from paper_recommender.vector_store import ExactVectorIndex


class RecommendRequest(BaseModel):
    url: str
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendationResponse(BaseModel):
    arxiv_id: str
    url: str
    primary_category: str
    categories: list[str]
    published_date: str | None
    updated_date: str | None
    similarity_score: float


class RecommendResponse(BaseModel):
    query_arxiv_id: str
    results: list[RecommendationResponse]


def create_app(db_path: str | Path = "data/paper_recommender.db", index_path: str | Path = "data/vectors.npz") -> FastAPI:
    app = FastAPI(title="Paper Recommender")
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/recommend", response_model=RecommendResponse)
    def recommend_endpoint(payload: RecommendRequest) -> RecommendResponse:
        try:
            arxiv_id = parse_arxiv_id(payload.url)
        except InvalidArxivUrl as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        conn = connect_db(db_path)
        try:
            index = ExactVectorIndex.load(index_path)
            results = recommend(
                conn,
                index,
                arxiv_id,
                top_k=payload.top_k,
                category=payload.category,
                date_from=payload.date_from,
                date_to=payload.date_to,
            )
        except RecommendationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
        finally:
            conn.close()

        return RecommendResponse(
            query_arxiv_id=arxiv_id,
            results=[
                RecommendationResponse(
                    arxiv_id=result.arxiv_id,
                    url=result.url,
                    primary_category=result.primary_category,
                    categories=list(result.categories),
                    published_date=result.published_date,
                    updated_date=result.updated_date,
                    similarity_score=result.similarity_score,
                )
                for result in results
            ],
        )

    return app


app = create_app()
```

- [ ] **Step 4: Run API tests to verify pass**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_app.py -v
```

Expected: all API tests PASS.

- [ ] **Step 5: Commit API**

```powershell
git add src/paper_recommender/app.py tests/test_app.py
git commit -m "Add recommendation API"
```

---

### Task 8: Minimal English UI

**Files:**
- Create: `src/paper_recommender/static/index.html`
- Create: `src/paper_recommender/static/app.js`
- Create: `src/paper_recommender/static/styles.css`
- Create: `tests/test_static_ui.py`

- [ ] **Step 1: Write static UI content test**

Create `tests/test_static_ui.py`:

```python
from pathlib import Path


def test_static_ui_uses_required_english_labels() -> None:
    html = Path("src/paper_recommender/static/index.html").read_text(encoding="utf-8")

    for label in [
        "arXiv URL",
        "Category",
        "Date range",
        "Top K",
        "Find similar papers",
        "Similar papers",
        "Open on arXiv",
        "No results",
    ]:
        assert label in html
```

- [ ] **Step 2: Run static UI test to verify failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_static_ui.py -v
```

Expected: FAIL because `index.html` does not exist.

- [ ] **Step 3: Add HTML**

Create `src/paper_recommender/static/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Paper Recommender</title>
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <main class="shell">
      <section class="search-panel">
        <h1>Paper Recommender</h1>
        <form id="recommend-form">
          <label>
            arXiv URL
            <input id="url" name="url" type="url" placeholder="https://arxiv.org/abs/1706.03762" required />
          </label>
          <div class="filters">
            <label>
              Category
              <input id="category" name="category" type="text" placeholder="cs.CL" />
            </label>
            <label>
              Date range
              <input id="date-from" name="date_from" type="date" />
              <input id="date-to" name="date_to" type="date" />
            </label>
            <label>
              Top K
              <input id="top-k" name="top_k" type="number" min="1" max="100" value="10" />
            </label>
          </div>
          <button type="submit">Find similar papers</button>
        </form>
      </section>

      <section class="results-panel">
        <h2>Similar papers</h2>
        <p id="status" class="status">No results</p>
        <ol id="results"></ol>
        <template id="result-template">
          <li class="result">
            <div>
              <strong class="paper-id"></strong>
              <span class="score"></span>
            </div>
            <div class="meta"></div>
            <a class="link" target="_blank" rel="noreferrer">Open on arXiv</a>
          </li>
        </template>
      </section>
    </main>
    <script src="/app.js"></script>
  </body>
</html>
```

- [ ] **Step 4: Add JavaScript**

Create `src/paper_recommender/static/app.js`:

```javascript
const form = document.querySelector("#recommend-form");
const statusNode = document.querySelector("#status");
const resultsNode = document.querySelector("#results");
const template = document.querySelector("#result-template");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusNode.textContent = "Searching...";
  resultsNode.replaceChildren();

  const formData = new FormData(form);
  const payload = {
    url: formData.get("url"),
    category: formData.get("category") || null,
    date_from: formData.get("date_from") || null,
    date_to: formData.get("date_to") || null,
    top_k: Number(formData.get("top_k") || 10),
  };

  const response = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    statusNode.textContent = body.detail || "No results";
    return;
  }

  if (body.results.length === 0) {
    statusNode.textContent = "No results";
    return;
  }

  statusNode.textContent = "";
  for (const result of body.results) {
    const item = template.content.cloneNode(true);
    item.querySelector(".paper-id").textContent = result.arxiv_id;
    item.querySelector(".score").textContent = result.similarity_score.toFixed(3);
    item.querySelector(".meta").textContent = `${result.primary_category} · ${result.published_date || "Unknown date"}`;
    item.querySelector(".link").href = result.url;
    resultsNode.appendChild(item);
  }
});
```

- [ ] **Step 5: Add CSS**

Create `src/paper_recommender/static/styles.css`:

```css
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f5f7fb;
  color: #172033;
}

body {
  margin: 0;
}

.shell {
  display: grid;
  grid-template-columns: minmax(280px, 420px) minmax(0, 1fr);
  gap: 24px;
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px;
}

.search-panel,
.results-panel {
  background: #ffffff;
  border: 1px solid #d8deea;
  border-radius: 8px;
  padding: 20px;
}

h1,
h2 {
  margin-top: 0;
}

form,
label,
.filters {
  display: grid;
  gap: 12px;
}

input,
button {
  min-height: 40px;
  border-radius: 6px;
  font: inherit;
}

input {
  border: 1px solid #b8c2d6;
  padding: 0 10px;
}

button {
  border: 0;
  background: #1f6feb;
  color: #ffffff;
  cursor: pointer;
}

#results {
  display: grid;
  gap: 12px;
  padding-left: 20px;
}

.result {
  padding: 12px;
  border: 1px solid #d8deea;
  border-radius: 8px;
}

.meta,
.score,
.status {
  color: #526079;
}

@media (max-width: 760px) {
  .shell {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 6: Run static UI test to verify pass**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_static_ui.py -v
```

Expected: static UI test PASS.

- [ ] **Step 7: Commit UI**

```powershell
git add src/paper_recommender/static/index.html src/paper_recommender/static/app.js src/paper_recommender/static/styles.css tests/test_static_ui.py
git commit -m "Add minimal English UI"
```

---

### Task 9: Sample Data Seed Script

**Files:**
- Create: `scripts/seed_sample_data.py`

- [ ] **Step 1: Add seed script**

Create `scripts/seed_sample_data.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from paper_recommender.models import Paper
from paper_recommender.storage import connect_db, init_db, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "paper_recommender.db"
INDEX_PATH = DATA_DIR / "vectors.npz"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    conn = connect_db(DB_PATH)
    init_db(conn)
    papers = [
        Paper("1706.03762", 1, True, "2024-01-02", "2017-06-12", "2023-08-02", "cs.CL", ("cs.CL", "cs.LG"), "sample-1"),
        Paper("1111.11111", 2, True, "2024-01-03", "2020-01-01", "2020-01-01", "cs.CL", ("cs.CL",), "sample-2"),
        Paper("2222.22222", 3, True, "2024-01-04", "2021-01-01", "2021-01-01", "cs.LG", ("cs.LG",), "sample-3"),
        Paper("3333.33333", 4, True, "2024-01-05", "2022-01-01", "2022-01-01", "stat.ML", ("stat.ML",), "sample-4"),
    ]
    for paper in papers:
        upsert_paper(conn, paper)
    conn.close()
    ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.2, 0.8, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    ).save(INDEX_PATH)
    print(f"Wrote {DB_PATH} and {INDEX_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full tests before manual seed**

Run:

```powershell
.\.venv\Scripts\python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run seed script**

Run:

```powershell
.\.venv\Scripts\python scripts/seed_sample_data.py
```

Expected: output includes `Wrote data\paper_recommender.db and data\vectors.npz`.

- [ ] **Step 4: Start local server**

Run:

```powershell
.\.venv\Scripts\python -m uvicorn paper_recommender.app:app --reload
```

Expected: server starts on `http://127.0.0.1:8000`.

- [ ] **Step 5: Verify API manually**

In a second terminal, run:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/recommend -ContentType 'application/json' -Body '{"url":"https://arxiv.org/abs/1706.03762","top_k":2}'
```

Expected: JSON response contains `query_arxiv_id` equal to `1706.03762` and results excluding `1706.03762`.

- [ ] **Step 6: Commit seed script**

```powershell
git add scripts/seed_sample_data.py
git commit -m "Add sample data seed script"
```

---

### Task 10: Final Verification

**Files:**
- Verify: all files created in Tasks 1-9

- [ ] **Step 1: Run complete test suite**

Run:

```powershell
.\.venv\Scripts\python -m pytest -v
```

Expected: every test passes.

- [ ] **Step 2: Run lint check**

Run:

```powershell
.\.venv\Scripts\python -m ruff check .
```

Expected: no lint errors.

- [ ] **Step 3: Verify no product naming regression**

Run:

```powershell
$patterns = @(
  'arxiv' + '-recommender',
  'arXiv Paper ' + 'Recommender',
  'arXiv paper recommendation ' + 'web service'
)
Get-ChildItem -Recurse -File |
  Where-Object { $_.FullName -notmatch '\\.git\\' } |
  Select-String -Pattern $patterns
```

Expected: no matches.

- [ ] **Step 4: Verify Git status**

Run:

```powershell
git status --short --branch
```

Expected: branch is `codex/paper-recommender-design` or a new implementation branch, and there are no uncommitted files.

- [ ] **Step 5: Push implementation branch**

Run:

```powershell
git push
```

Expected: branch pushes to `https://github.com/72-yh/paper-recommender.git`.

---

## Self-Review Checklist

Spec coverage in this plan:

- URL input and arXiv ID parsing: Task 2
- SQLite minimum metadata store: Task 3
- OAI normal/deleted record parsing: Task 4
- Content hash and re-embedding decision: Task 4
- One latest vector per paper in MVP storage: Tasks 3 and 5
- Self-exclusion, inactive exclusion, category/date filters: Task 6
- Clear English API errors: Tasks 2, 6, and 7
- English UI labels: Task 8
- Local file-based vector index: Task 5
- Sample end-to-end recommendation: Tasks 7 and 9

Design requirements reserved for follow-up plans:

- Production OAI backfill with network rate limiting
- Local open-source embedding model integration
- PCA/OPQ training and quantization
- Compressed ANN index quality gates
- Oracle Always Free or low-cost VPS deployment
- Backup and rollback automation

These are excluded from this MVP plan because they do not need to block the first verifiable user workflow.
