# retrieval-core

Chunking, embedding, storage and retrieval with citable provenance. Internal to
the portfolio — Minutebook, Knowhow and Overshoulder consume it by git pin.
**Never published to an index, never user-facing.**

> **Status: in progress.** Chunking, provenance and storage are built, including
> the deletion cascade. Embedding and dense retrieval are not yet. See
> [Build order](#build-order).

## Two rules that shape everything

**The library never opens a source file.** Products parse — PDFs, audio, screen
recordings, spreadsheets — and hand in [spans](src/retrieval_core/spans.py).
Parsing is where formats break, and keeping it in the product layer means a
Whisper or PDF failure costs one product rather than all three. Chunking moved
*in*; parsing deliberately did not.

**The provenance schema is frozen.** Everything else here can be broken freely
and the products re-pinned — the API is explicitly not a compatibility promise.
The schema in [`provenance.py`](src/retrieval_core/provenance.py) is different,
because it is held in place by indexes already written to users' disks. Changing
it forces a full reindex on their hardware, and you cannot patch that remotely.

## Using it

```python
from datetime import UTC, datetime
from retrieval_core import FileLocator, Source, SourceType, Span, chunk_spans

spans = [
    Span(text="Revenue rose 12% in Q3…", locator=FileLocator(0, 240, page=14)),
    Span(text="Costs were flat…", locator=FileLocator(240, 410, page=14)),
]

source = Source(
    source_id="report-2026-q3",
    source_type=SourceType.FILE,
    source_uri=r"C:\Shared\Finance\Q3.pdf",
    source_title="Q3.pdf",
    content_hash="sha256:…",
)

chunks = chunk_spans(spans, source, strategy="prose")
chunks[0].cite()  # 'Q3.pdf, p. 14'
```

## Chunking

Four strategies, targets all configurable via `ChunkingConfig`:

| Strategy | Target | Overlap | Hard rule |
|---|---|---|---|
| `prose` | ~400 tokens | ~15% | avoid splitting mid-paragraph |
| `transcript` | ~500 tokens | 1 turn | **never** split mid-turn |
| `steps` | 1 step | none | **never** merge across steps |
| `tabular` | header + 20 rows | header repeated | **never** split a row |

`chunk_size = min(model context, strategy target)` — **the model's context is a
ceiling, not a target.** Filling nomic v1.5's 8192-token window with one chunk
would dilute the embedding until nothing distinctive survives, destroy citation
granularity, and match many queries weakly instead of few strongly. The large
window is valuable because it means nothing is ever *forced* to fragment: a long
speaker turn or a verbose step can stay whole.

Overlap repeats **whole spans**, never slices of text. A half-paragraph overlap
would have to claim a locator it does not own.

### Locators are cut with the text

When a span is split, its locator is narrowed to the piece. When spans are
merged, their locators are merged — but only where a union is honest. Two
different pages, or two different sheets, keep the first locator rather than
inventing a range across them. A citation that sends somebody to the wrong
paragraph is worse than a coarse one.

Time locators interpolate by character position, which is an approximation and
known to be: the library is handed a turn with one start and one end, not
per-word timings. It lands within a second or two in ordinary speech. A parser
that has word timings should hand in finer spans rather than rely on it.

## Build order

1. ~~Provenance, locators, spans, chunking~~ — **done**
2. ~~Storage — SQLite + FTS5, memmapped vectors, deletion cascade~~ — **done**
3. **Embedding** — nomic-embed-text-v1.5 via ONNX Runtime, weights bundled
4. **Hybrid retrieval** — brute-force dense over NumPy, blended with FTS5

## Deletion actually deletes

`delete_by_source` is the most consequential method here. A purge that reports
success while leaving text searchable converts a privacy feature into a false
assurance, and a product cannot check from outside the library — so the library
guarantees it and `verify_deleted` proves it.

Four things go, and three are easy to get wrong:

- **chunk rows** — the obvious part
- **vectors, with the file compacted** — an orphaned row in the memmap keeps the
  embedding of deleted text, and embeddings are not anonymous
- **FTS5 including its shadow tables** — a plain `DELETE` leaves terms in FTS5's
  own b-tree where they stay readable in the file
- **`parent_id` links** in both the column *and* the provenance blob, since the
  blob is what every read deserialises

Then `VACUUM`, because SQLite frees pages rather than overwriting them — and a
**truncating WAL checkpoint**, because in WAL mode every write appends frames
that keep the old content until the log is truncated. That last step was found
by the raw-file test, not by reasoning: the database came back clean, every
query returned nothing, and the deleted text was sitting in `index.sqlite-wal`
in plain view.

Storage is NumPy plus the standard library's `sqlite3`. Not `sqlite-vec`: 204
open issues, no commits since May 2026, and wheels that bake the build machine's
AVX level and crash on older CPUs — unacceptable in a bundled binary shipped to
unknown hardware and unpatchable once installed.

## Tokenization

Chunking needs token counts; the real tokenizer arrives with the embedding
model. Until then `EstimatingCounter` stands in behind the same interface. It
rounds **up** deliberately: an under-count could push a chunk past the model's
ceiling and have it silently truncated, losing text the index claims is there.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
