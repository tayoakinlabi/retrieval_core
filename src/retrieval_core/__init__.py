"""retrieval-core — chunking, embedding, storage and retrieval with citable provenance.

Internal to the portfolio and never user-facing. Consumed by Minutebook, Knowhow
and Overshoulder by git pin, never published to an index (L1).

Two things here are load-bearing and worth knowing before changing anything:

1. **The provenance schema is frozen** (contract §2). Everything else may be
   broken freely and the products re-pinned; that one cannot, because it is held
   in place by indexes already written to users' disks, and a change forces a
   full reindex on their hardware.
2. **The library never opens a source file** (§1.2). Products parse and hand in
   spans. Parsing is where formats break, and quarantining it in the product
   layer means one broken parser costs one product rather than all three.
"""

from __future__ import annotations

from retrieval_core.chunking import (
    STRATEGIES,
    Chunk,
    ChunkingConfig,
    Source,
    chunk_spans,
)
from retrieval_core.provenance import (
    CellLocator,
    Channel,
    FileLocator,
    Locator,
    ProvenanceRecord,
    SourceType,
    StepLocator,
    TimeLocator,
)
from retrieval_core.spans import Span, SpanKind
from retrieval_core.storage import (
    Collection,
    CollectionStats,
    IndexMismatchError,
    SourceSummary,
)
from retrieval_core.tokens import EstimatingCounter, TokenCounter

__version__ = "0.1.0"

__all__ = [
    "STRATEGIES",
    "CellLocator",
    "Channel",
    "Collection",
    "CollectionStats",
    "Chunk",
    "ChunkingConfig",
    "EstimatingCounter",
    "FileLocator",
    "IndexMismatchError",
    "Locator",
    "ProvenanceRecord",
    "Source",
    "SourceSummary",
    "SourceType",
    "Span",
    "SpanKind",
    "StepLocator",
    "TimeLocator",
    "TokenCounter",
    "chunk_spans",
]
