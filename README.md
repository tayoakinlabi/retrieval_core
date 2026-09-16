# retrieval-core

Chunking, embedding, storage and retrieval with citable provenance. Internal to
the portfolio — Minutebook, Knowhow and Overshoulder consume it by git pin.
**Never published to an index, never user-facing.**

> **Status: in progress.** Chunking, provenance, storage and embedding are
> built, including the deletion cascade. Hybrid retrieval is not yet. See
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
ceiling, not a target.** Filling the window with one chunk would dilute the
embedding until nothing distinctive survives, destroy citation granularity, and
match many queries weakly instead of few strongly. The window matters because it
means nothing is *forced* to fragment: a long speaker turn or a verbose step can
stay whole. At 2048 usable tokens (see [Embedding](#embedding)) that covers a
turn of roughly 1,500 words, which is longer than anything the three products
have produced so far.

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
3. ~~Embedding — nomic-embed-text-v1.5 via ONNX Runtime~~ — **done**
4. **Hybrid retrieval** — brute-force dense over NumPy, blended with FTS5

## Embedding

nomic-embed-text-v1.5, INT8 ONNX, on the CPU. Always local — the API-key path in
the portfolio covers generation only, so a lapsed key never renders an index
unusable and collections stay re-indexable offline.

```python
from retrieval_core.embedding import NomicEmbedder

embedder = NomicEmbedder.load()  # add allow_download=True on a fresh checkout
vectors = embedder.embed_documents([chunk.text for chunk in chunks])
query = embedder.embed_query("When is the VAT return due?")
scores = vectors @ query  # unit vectors, so this is cosine
```

`onnxruntime` and `tokenizers` are the `embedding` extra, not base dependencies.
Chunking and storage work without them, and `import retrieval_core` does not pull
them in — a product re-indexing from a cached vector file should not load a
60 MB runtime to do it.

### Three things that go wrong silently

**The task prefixes are internal.** Nomic wants `search_document:` on indexed
text and `search_query:` on queries. There are two methods, not one method with a
flag, and no way to spell either wrong — a flag is exactly how a default gets
inverted somewhere up the stack with nothing failing and nothing logging.

**Truncate before normalising.** The order is mean-pool → layer-norm →
*truncate* → L2-normalise. Matryoshka works because the leading dimensions carry
the signal, but the first 256 components of a unit vector are not themselves a
unit vector. Truncating afterwards leaves vectors of varying length, a dot
product stops being a cosine, and results reorder by how much of each vector's
mass happened to land in the leading dimensions.

**Padding is excluded from the mean**, not averaged in as zeros — otherwise a
short text's vector is pulled toward the origin in proportion to how much
padding it happened to receive.

### Two things measured here that the contract did not expect

**The usable context is 2048 tokens, not 8192.** The checkpoint's `config.json`
reports `max_trained_positions: 2048` with no rotary scaling factor set, so
positions beyond that are untrained. The export will still *run* longer inputs —
but a single 8192-token forward pass took **949 seconds** on this machine against
3.9 s at 4096 and 1.4 s at 2048. The model spec pins 2048, and `TokenizerCounter`
subtracts the prefix and special tokens on top, so chunks are sized against a
ceiling the text can actually occupy.

**Batching changes the vectors, so batching is off by default.** Two different
texts in one forward pass move each other's hidden states by around 10%; the
same text twice changes nothing, bit for bit. That is dynamic quantisation — the
INT8 graph derives activation scales from each tensor's range at runtime. It
survives pooling as roughly 0.985 cosine between a text embedded alone and
embedded beside others.

0.985 sounds harmless and is not. This model was chosen over BGE Small precisely
because same-domain similarity fell from 0.93–0.97 to 0.46–0.70, which is what
made vector dedup usable; a 0.985 noise floor set by batch composition sits above
that entire band. It also breaks reproducibility — re-indexing a corpus would
produce vectors that no longer match their neighbours from the first pass.

So `batch_size` defaults to 1. Measured cost: 71 texts/s against 135 at batch 16,
which is under two minutes per 10,000 chunks either way for a one-off job.
`test_the_int8_export_is_still_batch_sensitive` is a canary — if a future export
fixes this, that test fails and the default can go back up.

### Weights

137 MB, so they cannot live in git. Products bundle them in the installer;
a checkout downloads them once into `%LOCALAPPDATA%\retrieval-core\models`.

```python
NomicEmbedder.load(allow_download=True)  # once, then never again
```

`resolve()` looks in the product's bundled directory first, then the cache, and
**refuses to reach the network unless asked** — an offline install that quietly
started downloading a model would break the privacy claim on the one machine
least able to notice. Every file is SHA-256 checked on every load, bundled copies
included, against a pinned commit rather than `main`. Not because downloads fail
loudly, but because they fail quietly: wrong weights do not crash, they produce
embeddings that are merely wrong, and the index looks completely normal for as
long as it lives.

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

Chunking needs token counts. `EstimatingCounter` needs no model and rounds **up**
deliberately — an under-count could push a chunk past the model's ceiling and
have it silently truncated, losing text the index claims is there.

With weights available, `embedder.token_counter` is the model's own tokenizer
behind the same interface, and its `max_context` is the model's ceiling *minus*
the task prefix and special tokens. The caller's text is never what reaches the
model, and a chunk sized against the raw ceiling loses its tail.

```python
chunks = chunk_spans(spans, source, counter=embedder.token_counter)
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
