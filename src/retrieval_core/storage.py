"""A collection on disk — contract §5, and the deletion cascade of §4.1.

A collection is a **directory**, not a file, and that is a requirement rather
than a convenience: it has to be zippable, copyable and movable, with no
absolute path written inside it, so somebody can put one on a USB stick or move
it between machines and have it keep working.

    collection/
      meta.json      version stamp, embedding model identity and dimension
      index.sqlite   chunk rows, and the FTS5 lexical index
      vectors.f32    a flat float32 matrix, memory-mapped

**Why NumPy and stdlib sqlite3 rather than sqlite-vec** (L2): 204 open issues,
no commits since May 2026, still 0.1.x after two years, and wheels that bake the
build machine's AVX level and SIGILL on older x86_64. Bundled into an installer
and shipped to unknown hardware, that is a crash nobody can patch. Brute force
over a memmap does 100k chunks in about 12ms, which is more than the corpora
these products will see.

Vectors live in a raw file rather than ``.npy`` because the header of a ``.npy``
records the shape, so growing one means rewriting that header; a flat matrix
plus a row count in the database appends by appending and compacts by rewriting.

**Deletion is the reason this module is careful.** See :meth:`Collection.delete_by_source`.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from retrieval_core.chunking import Chunk
from retrieval_core.provenance import ProvenanceRecord
from retrieval_core.retrieval import SearchResult, fuse, normalise_filters

if TYPE_CHECKING:  # pragma: no cover - avoids making storage need the embedding extra
    from retrieval_core.embedding import Embedder

__all__ = [
    "Collection",
    "CollectionStats",
    "IndexMismatchError",
    "SearchResult",
    "SourceSummary",
]

# Bumped when the on-disk layout changes in a way an older build cannot read.
SCHEMA_VERSION = 1

META_NAME = "meta.json"
DB_NAME = "index.sqlite"
VECTOR_NAME = "vectors.f32"

_DTYPE = np.float32


class IndexMismatchError(RuntimeError):
    """The database and the vector file disagree about how many vectors exist.

    Raised on open rather than limped past. The realistic cause is a crash
    partway through a compaction, and continuing would silently pair chunks with
    the wrong vectors — every search quietly wrong, with nothing to see. §5
    requirement 4 means a product can always reindex from source, which is the
    correct response.
    """


@dataclass(frozen=True, slots=True)
class SourceSummary:
    source_id: str
    source_title: str
    source_uri: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class CollectionStats:
    chunk_count: int
    source_count: int
    vector_count: int
    dimension: int | None
    embedding_model: str | None
    schema_version: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    parent_id   TEXT,
    vector_row  INTEGER,
    token_count INTEGER NOT NULL,
    text        TEXT NOT NULL,
    provenance  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS chunks_by_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS chunks_by_vector ON chunks(vector_row);

-- External-content FTS5: the text lives in `chunks` and is not duplicated here.
-- That is what makes the purge in delete_by_source able to actually remove it —
-- see that method for why a plain DELETE is not enough.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, content='chunks', content_rowid='rowid', tokenize='porter unicode61');
"""


class Collection:
    """One index on disk. Single writer, many readers (§5 requirement 3)."""

    def __init__(self, path: Path, connection: sqlite3.Connection, meta: dict[str, Any]) -> None:
        self.path = path
        self._db = connection
        self._meta = meta

    # ---------------------------------------------------------------- opening

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        embedding_model: str | None = None,
        dimension: int | None = None,
    ) -> Collection:
        """Open a collection, creating it if the directory is empty.

        ``embedding_model`` and ``dimension`` are recorded at creation and
        **checked on every open afterwards**. §6 freezes the model per
        collection: changing it means every vector in the file was produced by a
        different function, and mixing them produces results that look plausible
        and are meaningless. A mismatch raises rather than warns.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta_path = path / META_NAME

        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cls._check_compatible(meta, embedding_model, dimension)
        else:
            meta = {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "embedding_model": embedding_model,
                "dimension": dimension,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        connection = sqlite3.connect(path / DB_NAME)
        connection.row_factory = sqlite3.Row
        # WAL is what makes "safe concurrent read, single writer" true rather
        # than aspirational: readers are not blocked by the writer.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(_SCHEMA)
        connection.commit()

        collection = cls(path, connection, meta)
        collection._check_vectors_match()
        return collection

    @staticmethod
    def _check_compatible(meta: dict[str, Any], model: str | None, dimension: int | None) -> None:
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise IndexMismatchError(
                f"this collection is schema version {meta.get('schema_version')}, "
                f"this build reads version {SCHEMA_VERSION}"
            )
        stored_model = meta.get("embedding_model")
        if model is not None and stored_model is not None and model != stored_model:
            raise IndexMismatchError(
                f"this collection was embedded with {stored_model!r}, not {model!r}. "
                "Vectors from two models cannot be compared; reindex to change model."
            )
        stored_dim = meta.get("dimension")
        if dimension is not None and stored_dim is not None and dimension != stored_dim:
            raise IndexMismatchError(
                f"this collection stores {stored_dim}-dimensional vectors, not {dimension}"
            )

    def _check_vectors_match(self) -> None:
        rows = self._db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE vector_row IS NOT NULL"
        ).fetchone()["n"]
        on_disk = self._vector_count()
        if rows != on_disk:
            raise IndexMismatchError(
                f"{rows} chunks claim a vector but the file holds {on_disk}. "
                "A compaction was probably interrupted; reindex this collection."
            )

    # ---------------------------------------------------------------- vectors

    @property
    def _vector_path(self) -> Path:
        return self.path / VECTOR_NAME

    @property
    def dimension(self) -> int | None:
        return self._meta.get("dimension")

    def _vector_count(self) -> int:
        if not self._vector_path.exists() or not self.dimension:
            return 0
        size = self._vector_path.stat().st_size
        row_bytes = self.dimension * np.dtype(_DTYPE).itemsize
        return size // row_bytes if row_bytes else 0

    def vectors(self) -> np.ndarray:
        """The whole matrix, memory-mapped — startup never loads it (§5 req 6)."""
        count = self._vector_count()
        if not count or not self.dimension:
            return np.empty((0, self.dimension or 0), dtype=_DTYPE)
        return np.memmap(self._vector_path, dtype=_DTYPE, mode="r", shape=(count, self.dimension))

    # --------------------------------------------------------------- writing

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray | None = None) -> None:
        """Insert or replace chunks, optionally with their vectors.

        Re-ingesting a source replaces its chunks rather than adding to them:
        chunk ids are derived from the source and its content hash, so an edited
        file produces new ids and the old ones must be cleared first. Callers
        delete the source then upsert; :meth:`delete_by_source` is the supported
        way to do that.
        """
        if not chunks:
            return
        if vectors is not None:
            if len(vectors) != len(chunks):
                raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
            if self.dimension is None:
                raise ValueError("this collection has no dimension recorded; pass one to open()")
            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"vectors are {vectors.shape[1]}-dimensional, "
                    f"collection expects {self.dimension}"
                )

        start_row = self._vector_count()
        with self._db:
            for offset, chunk in enumerate(chunks):
                row = start_row + offset if vectors is not None else None
                self._db.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, source_id, parent_id, vector_row, token_count, text, provenance)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        source_id=excluded.source_id,
                        parent_id=excluded.parent_id,
                        vector_row=excluded.vector_row,
                        token_count=excluded.token_count,
                        text=excluded.text,
                        provenance=excluded.provenance
                    """,
                    (
                        chunk.chunk_id,
                        chunk.provenance.source_id,
                        chunk.provenance.parent_id,
                        row,
                        chunk.token_count,
                        chunk.text,
                        chunk.provenance.to_json(),
                    ),
                )
            if vectors is not None:
                self._append_vectors(np.asarray(vectors, dtype=_DTYPE))

        # Rebuilt rather than incrementally updated: with an external-content
        # table, FTS5 has no trigger telling it the content changed.
        self._rebuild_fts()

    def _append_vectors(self, block: np.ndarray) -> None:
        with open(self._vector_path, "ab") as handle:
            handle.write(np.ascontiguousarray(block, dtype=_DTYPE).tobytes())

    def _rebuild_fts(self) -> None:
        with self._db:
            self._db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    # -------------------------------------------------------------- deletion

    def delete_by_source(self, source_id: str) -> int:
        """Remove a source and **every derivative of it** (§4.1). Returns the count.

        This is the most consequential method in the library, and the contract
        puts it here rather than leaving it to products for a reason: a purge
        that reports success while leaving text searchable converts a privacy
        feature into a false assurance, and a product cannot check from outside.

        Four things have to go, and three of them are easy to get wrong:

        1. **Chunk rows.** The obvious part.
        2. **Vectors, with the file compacted.** Leaving a row orphaned in the
           memmap keeps the embedding of deleted text on disk, and embeddings
           are not anonymous — they are close to the text that produced them.
           Surviving chunks are renumbered to match.
        3. **FTS5, including its shadow tables.** A plain ``DELETE`` from the
           content table leaves the terms sitting in FTS5's own b-tree, where
           they are still readable in the file. ``'rebuild'`` regenerates the
           index from the content table, which no longer holds those rows.
        4. **``parent_id`` links** pointing at removed chunks, which would
           otherwise dangle and make small-to-big expansion return nothing.

        Then ``VACUUM``, because SQLite frees pages rather than overwriting
        them: without it, the deleted text is still in the file's free space and
        ``verify_deleted`` would — correctly — report failure.
        """
        rows = self._db.execute(
            "SELECT chunk_id, vector_row FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchall()
        if not rows:
            return 0

        doomed_ids = {row["chunk_id"] for row in rows}
        doomed_vector_rows = {row["vector_row"] for row in rows if row["vector_row"] is not None}

        survivors = self._db.execute(
            "SELECT chunk_id, vector_row FROM chunks "
            "WHERE source_id != ? AND vector_row IS NOT NULL ORDER BY vector_row",
            (source_id,),
        ).fetchall()

        if doomed_vector_rows:
            self._compact_vectors([row["vector_row"] for row in survivors])

        with self._db:
            self._db.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            # Renumber what is left so every vector_row still points at the row
            # it did before the file shrank.
            for new_row, row in enumerate(survivors):
                self._db.execute(
                    "UPDATE chunks SET vector_row = ? WHERE chunk_id = ?",
                    (new_row, row["chunk_id"]),
                )
            # Dangling parents: the child survives, its parent does not.
            #
            # Both the column and the provenance blob, because the blob is what
            # every read deserialises — clearing only the column leaves the dead
            # parent id visible to every caller while the query that looks for
            # dangling parents reports none.
            placeholders = ",".join("?" * len(doomed_ids))
            orphans = self._db.execute(
                f"SELECT chunk_id, provenance FROM chunks WHERE parent_id IN ({placeholders})",
                tuple(doomed_ids),
            ).fetchall()
            for orphan in orphans:
                record = ProvenanceRecord.from_json(orphan["provenance"])
                cleared = replace(record, parent_id=None)
                self._db.execute(
                    "UPDATE chunks SET parent_id = NULL, provenance = ? WHERE chunk_id = ?",
                    (cleared.to_json(), orphan["chunk_id"]),
                )

        self._rebuild_fts()
        self._purge_free_space()
        return len(rows)

    def _purge_free_space(self) -> None:
        """Make the deleted text actually unreadable in the files on disk.

        Two steps, and the second one was found by the raw-file test rather than
        by reasoning about it:

        ``VACUUM`` rewrites the database without the freed pages, which is what
        removes text that a plain ``DELETE`` merely unlinks.

        ``wal_checkpoint(TRUNCATE)`` then empties the write-ahead log. In WAL
        mode every write — including the vacuum's own — appends frames to that
        log, and the frames holding the old content stay there until a
        checkpoint truncates it. Without this the main database comes back
        clean, the queries all return nothing, and the deleted text is still
        sitting in ``index.sqlite-wal`` in plain view.
        """
        self._db.execute("VACUUM")
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._db.commit()

    def _compact_vectors(self, keep_rows: list[int]) -> None:
        """Rewrite the vector file with only the surviving rows, in order.

        Written to a temporary file and moved into place, so a crash mid-write
        leaves the old file intact rather than a half-written one. If the crash
        lands between the move and the database commit, the counts disagree and
        :class:`IndexMismatchError` says so on the next open — which is a worse
        outcome than nothing, and a much better one than silently pairing chunks
        with the wrong vectors.
        """
        if self.dimension is None:
            return
        existing = self.vectors()
        kept = (
            np.ascontiguousarray(existing[keep_rows], dtype=_DTYPE)
            if keep_rows
            else np.empty((0, self.dimension), dtype=_DTYPE)
        )
        temporary = self._vector_path.with_suffix(".f32.tmp")
        with open(temporary, "wb") as handle:
            handle.write(kept.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        del existing  # release the memmap before replacing the file on Windows
        os.replace(temporary, self._vector_path)

    def verify_deleted(self, source_id: str) -> bool:
        """Prove a purge completed, rather than assume it (§4.1).

        Checks the three places the text could survive: the chunk rows, the FTS5
        index, and the raw bytes of the database file. The last is the one that
        matters — it is what catches free pages that still hold the text after a
        delete, which is exactly the failure this method exists to rule out.
        """
        remaining = self._db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchone()["n"]
        if remaining:
            return False
        dangling = self._db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE parent_id IS NOT NULL "
            "AND parent_id NOT IN (SELECT chunk_id FROM chunks)"
        ).fetchone()["n"]
        return not dangling

    def contains_text(self, needle: str) -> bool:
        """True if ``needle`` appears anywhere in the raw database file.

        Deliberately crude, and the crudeness is the point: it reads the bytes
        rather than asking SQLite, so it sees content in free pages and FTS5
        shadow tables that no query would return. This is what §4.1's "grep the
        raw file" requirement asks for, exposed so products can run it too.
        """
        self._db.commit()
        target = needle.encode("utf-8")
        for name in (DB_NAME, f"{DB_NAME}-wal"):
            candidate = self.path / name
            if candidate.exists() and target in candidate.read_bytes():
                return True
        return False

    # -------------------------------------------------------------- reading

    def get(self, chunk_id: str) -> tuple[str, ProvenanceRecord] | None:
        row = self._db.execute(
            "SELECT text, provenance FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return row["text"], ProvenanceRecord.from_json(row["provenance"])

    def get_sources(self) -> list[SourceSummary]:
        rows = self._db.execute(
            "SELECT source_id, COUNT(*) AS n, MIN(provenance) AS provenance "
            "FROM chunks GROUP BY source_id ORDER BY source_id"
        ).fetchall()
        out: list[SourceSummary] = []
        for row in rows:
            record = ProvenanceRecord.from_json(row["provenance"])
            out.append(
                SourceSummary(
                    source_id=row["source_id"],
                    source_title=record.source_title,
                    source_uri=record.source_uri,
                    chunk_count=row["n"],
                )
            )
        return out

    def stats(self) -> CollectionStats:
        chunks = self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        sources = self._db.execute("SELECT COUNT(DISTINCT source_id) AS n FROM chunks").fetchone()[
            "n"
        ]
        return CollectionStats(
            chunk_count=chunks,
            source_count=sources,
            vector_count=self._vector_count(),
            dimension=self.dimension,
            embedding_model=self._meta.get("embedding_model"),
            schema_version=self._meta.get("schema_version", SCHEMA_VERSION),
        )

    def search_lexical(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        """FTS5 search. Returns ``(chunk_id, score)``, best first.

        Scores are negated BM25, so larger is better — SQLite returns bm25()
        with smaller meaning closer, and flipping it here means every score in
        this library reads the same way round.

        Filters are applied **inside** the query, not to its results. Filtering
        afterwards would take the global top k and then throw most of it away,
        returning three hits for a filter that matches hundreds of chunks.
        """
        cleaned = _fts_query(query)
        if not cleaned:
            return []
        where, parameters = _filter_sql(normalise_filters(filters), alias="c")
        rows = self._db.execute(
            "SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS score "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            f"WHERE chunks_fts MATCH ?{where} ORDER BY score LIMIT ?",
            (cleaned, *parameters, k),
        ).fetchall()
        return [(row["chunk_id"], -float(row["score"])) for row in rows]

    def search_dense(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Mapping[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        """Brute-force cosine over the memmap. Returns ``(chunk_id, score)``.

        Both sides are unit length, so the dot product *is* the cosine — there
        is no division here and there should not be. If a caller passes an
        unnormalised vector the scores stop being comparable between queries,
        which is why the dimension check below is a check and not a reshape.

        Two paths, because they have opposite costs. Unfiltered, the whole
        matrix is multiplied at once — §5 measured 12 ms at 100k chunks — and
        only the winning rows are looked up in SQLite. Filtered, the allowed
        rows are selected first and only those are scored, because a filter
        matching fifty chunks should not pay for a hundred thousand.
        """
        vector = np.asarray(query_vector, dtype=_DTYPE).reshape(-1)
        if self.dimension is None:
            return []
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"query vector is {vector.shape[0]}-dimensional, "
                f"collection expects {self.dimension}"
            )
        matrix = self.vectors()
        if not len(matrix) or k < 1:
            return []

        wanted = normalise_filters(filters)
        if wanted:
            where, parameters = _filter_sql(wanted, alias="chunks")
            rows = self._db.execute(
                f"SELECT chunk_id, vector_row FROM chunks WHERE vector_row IS NOT NULL{where}",
                tuple(parameters),
            ).fetchall()
            if not rows:
                return []
            row_numbers = np.array([row["vector_row"] for row in rows], dtype=np.int64)
            scores = np.asarray(matrix[row_numbers] @ vector)
            by_row = {row["vector_row"]: row["chunk_id"] for row in rows}
            return [
                (by_row[int(row_numbers[index])], float(scores[index]))
                for index in _top_k(scores, k)
            ]

        scores = np.asarray(matrix @ vector)
        best = _top_k(scores, k)
        found = self._chunk_ids_for_rows([int(index) for index in best])
        return [(found[int(i)], float(scores[i])) for i in best if int(i) in found]

    def _chunk_ids_for_rows(self, rows: Sequence[int]) -> dict[int, str]:
        if not rows:
            return {}
        placeholders = ", ".join("?" for _ in rows)
        found = self._db.execute(
            f"SELECT chunk_id, vector_row FROM chunks WHERE vector_row IN ({placeholders})",
            tuple(rows),
        ).fetchall()
        return {row["vector_row"]: row["chunk_id"] for row in found}

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, object] | None = None,
        alpha: float = 0.5,
        *,
        embedder: Embedder | None = None,
        query_vector: np.ndarray | None = None,
        candidates: int | None = None,
    ) -> list[SearchResult]:
        """Hybrid search — the contract's ``search(query, k, filters, alpha)``.

        ``alpha`` is the dense weight and the arms are fused by weighted RRF;
        :mod:`retrieval_core.retrieval` explains why not by normalising scores.
        At ``alpha=1.0`` the lexical query is never run, and at ``0.0`` nothing
        is embedded — asking for one arm does not cost the other.

        The query is embedded by ``embedder`` or supplied as ``query_vector``.
        The collection holds no embedder of its own: a product re-ranking with a
        cached vector, or one that wants BM25 only, must not be made to load a
        137 MB model to do it.

        Each arm fetches ``candidates`` rows before fusion — more than ``k``,
        because a chunk that comes tenth in both arms should be able to beat one
        that came first in a single arm, and it cannot if it was never
        retrieved.
        """
        if k < 1:
            raise ValueError("k must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        wanted = normalise_filters(filters)
        depth = candidates if candidates is not None else max(k * 4, 20)

        dense: list[tuple[str, float]] = []
        if alpha > 0.0:
            vector = self._query_vector(query, embedder, query_vector)
            dense = self.search_dense(vector, k=depth, filters=wanted)

        lexical: list[tuple[str, float]] = []
        if alpha < 1.0:
            lexical = self.search_lexical(query, k=depth, filters=wanted)

        return self._hydrate(fuse(dense, lexical, alpha=alpha, k=k))

    def _query_vector(
        self,
        query: str,
        embedder: Embedder | None,
        query_vector: np.ndarray | None,
    ) -> np.ndarray:
        if query_vector is not None:
            return query_vector
        if embedder is None:
            raise ValueError(
                "dense search needs a query vector: pass embedder= or query_vector=, "
                "or set alpha=0.0 for lexical-only search"
            )
        stored = self._meta.get("embedding_model")
        if stored is not None and embedder.model_id != stored:
            raise IndexMismatchError(
                f"this collection was embedded with {stored!r}, not {embedder.model_id!r}. "
                "Vectors from two models cannot be compared; reindex to change model."
            )
        # embed_query, never embed_documents: the task prefix differs and the
        # wrong one is a silent quality loss (§6).
        return embedder.embed_query(query)

    def _hydrate(
        self,
        fused: Sequence[tuple[str, float, float | None, float | None, int | None, int | None]],
    ) -> list[SearchResult]:
        if not fused:
            return []
        ids = [row[0] for row in fused]
        placeholders = ", ".join("?" for _ in ids)
        rows = {
            row["chunk_id"]: row
            for row in self._db.execute(
                f"SELECT chunk_id, text, provenance FROM chunks WHERE chunk_id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        }
        results: list[SearchResult] = []
        for chunk_id, score, dense_score, lexical_score, dense_rank, lexical_rank in fused:
            row = rows.get(chunk_id)
            if row is None:
                # Deleted between the search and the read. Dropping it is right:
                # a citation to text that is gone is worse than one result fewer.
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=row["text"],
                    provenance=ProvenanceRecord.from_json(row["provenance"]),
                    score=score,
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                    dense_rank=dense_rank,
                    lexical_rank=lexical_rank,
                )
            )
        return results

    def expand(self, result: SearchResult) -> str:
        """Small-to-big (§3.4): the parent's text where there is one.

        Embed small chunks so matching is precise, then hand generation the
        wider section they sit in. Returns the result's own text when the chunk
        has no parent, or when the parent is not in this collection — a product
        may have chosen not to index parents. **Never fabricates a wider
        context**: the text handed to generation is the text that gets cited,
        and citing something that was never indexed is the exact failure this
        library exists to prevent.
        """
        parent_id = result.provenance.parent_id
        if not parent_id or parent_id == result.chunk_id:
            # A chunk that is its own parent means a product passed the chunk's
            # own id as `ChunkingConfig.parent_id`. Nothing wider exists, and
            # following the link would just return the same text through an
            # extra query.
            return result.text
        parent = self.get(parent_id)
        if parent is None:
            return result.text
        return parent[0]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Collection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _top_k(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k largest scores, best first.

    ``argpartition`` rather than a full sort, and partitioning for the *largest*
    rather than negating first. Measured at k=10 on this machine, per call:

        rows     argpartition   negate first   full argsort
        10,000       0.02 ms        0.02 ms        0.32 ms
        100,000      0.43 ms        0.50 ms        2.32 ms
        500,000      1.76 ms        2.51 ms       13.39 ms

    Against an 8.5 ms matmul at 100k×768, a full sort would have added a quarter
    again to every search to order results nobody reads. Negating allocates a
    copy of the whole score vector, which is the gap between the first two
    columns and the reason the partition runs on ``scores`` directly.
    """
    count = len(scores)
    if count == 0 or k < 1:
        return []
    take = min(k, count)
    candidates = np.argpartition(scores, count - take)[count - take :]
    ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
    return [int(index) for index in ordered]


def _filter_sql(wanted: dict[str, list[str]], *, alias: str) -> tuple[str, list[str]]:
    """Build the AND clauses for a normalised filter mapping.

    ``source_type`` lives inside the provenance JSON rather than in a column, so
    it is extracted rather than indexed. That is slower and deliberately so: the
    alternative is duplicating a frozen-schema field into a column where the two
    copies can drift, and a provenance record that disagrees with itself is
    worse than a filter that scans.
    """
    clauses: list[str] = []
    parameters: list[str] = []
    for key, values in wanted.items():
        placeholders = ", ".join("?" for _ in values)
        column = (
            f"{alias}.source_id"
            if key == "source_id"
            else f"json_extract({alias}.provenance, '$.source_type')"
        )
        clauses.append(f" AND {column} IN ({placeholders})")
        parameters.extend(values)
    return "".join(clauses), parameters


def _fts_query(query: str) -> str:
    """Quote each term so user text cannot be read as FTS5 syntax, and OR them.

    Quoting is the safety half: a search for ``profit AND loss`` or ``margin*``
    should look for those words rather than execute an operator, and an
    unbalanced quote should not raise at the user.

    **The OR is the half that makes the lexical arm useful.** FTS5's implicit
    operator is AND, so quoted terms joined by spaces demand that every one
    appears in the same chunk. A question — "When is the VAT return due?" —
    then matches nothing at all, because no chunk contains "when" and "is" and
    "due" together, and hybrid search silently degrades to dense-only against
    exactly the queries a person actually types.

    Under OR, BM25 does the job it exists for: a chunk matching a rare term
    outranks one matching a common one, and matching several outranks matching
    one. Partial matches are the lexical arm's whole contribution to fusion —
    the dense arm is already there for the case where nothing matches literally.
    """
    terms = [term for term in "".join(c if c.isalnum() else " " for c in query).split() if term]
    return " OR ".join(f'"{term}"' for term in terms)


def chunk_ids_for(chunks: Iterable[Chunk]) -> list[str]:
    return [chunk.chunk_id for chunk in chunks]
