"""Blending two rankings that do not share a scale.

Dense search returns cosine similarity — bounded, comparable between queries,
0.72 means the same thing today as tomorrow. BM25 returns a number that depends
on the corpus, the term frequencies in it, and the length of the document; it
has no ceiling and no fixed meaning. Adding them is meaningless, and every
obvious repair is worse than it looks:

**Min-max normalising each side first** is the common answer and it fabricates
confidence. Normalising rescales whatever it is handed to fill [0, 1], so the
best of ten mediocre hits becomes 1.0 — identical to the best of ten excellent
ones. A query that matched nothing then produces a top result that scores like a
perfect match, and the caller has no way to tell the difference. Z-scoring has
the same defect and adds instability: at k=10 the standard deviation is being
estimated from ten points.

**So the default is weighted Reciprocal Rank Fusion**, which uses only the
positions. A document's contribution is ``weight / (rrf_k + rank)``, summed over
the arms that found it. Nothing is rescaled, nothing depends on the distribution
of the candidate set, and the fused ordering is the same whether BM25 returned
scores around 4 or around 400.

What RRF gives up is magnitude: a dense hit at 0.95 and one at 0.55 both rank
first and both contribute the same. That is a real loss, and it is why
:class:`SearchResult` carries ``dense_score`` and ``lexical_score`` alongside the
fused one. **Threshold on those, never on ``score``** — the fused number is a
ranking quantity with no units, and a caller who filters on ``score > 0.5`` is
filtering on rank position, not on relevance. The dataclass says so and so does
this sentence, because it is the mistake this design invites.

``alpha`` is the dense weight, matching the contract's ``search(query, k,
filters, alpha)``. It degenerates honestly: at 1.0 the lexical query is not run
at all, and at 0.0 neither is the embedding, so a caller who wants one arm does
not pay for the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retrieval_core.provenance import ProvenanceRecord

__all__ = [
    "RRF_K",
    "SearchResult",
    "fuse",
]

# The constant from the original RRF paper. It damps the difference between the
# top ranks: without it, rank 1 would be worth twice rank 2, which lets a single
# arm's confident mistake beat two arms' agreement. At 60 the top few ranks are
# close together and the ordering is decided by how many arms found a document
# at all, which is the property that makes fusion better than either arm.
RRF_K = 60


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One hit, with the fused ranking and both arms' own scores.

    ``score`` is the fused RRF value. It is **not** a similarity and not
    comparable between queries — use it to order results and nothing else.

    ``dense_score`` is cosine similarity and *is* comparable between queries;
    ``lexical_score`` is negated BM25, so larger is better but the scale depends
    on the corpus. Either is ``None`` when that arm did not return this chunk —
    which is information, not a gap: it means the other arm found something this
    one missed.
    """

    chunk_id: str
    text: str
    provenance: ProvenanceRecord
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None

    def cite(self) -> str:
        return self.provenance.cite()

    @property
    def found_by_both(self) -> bool:
        """Both arms returned this chunk — the strongest signal fusion produces."""
        return self.dense_score is not None and self.lexical_score is not None


def fuse(
    dense: Sequence[tuple[str, float]],
    lexical: Sequence[tuple[str, float]],
    *,
    alpha: float = 0.5,
    k: int = 10,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float, float | None, float | None, int | None, int | None]]:
    """Weighted RRF over two ranked lists of ``(chunk_id, score)``.

    Both inputs must already be ordered best-first, which is what every caller
    in this library produces. Returns ``(chunk_id, fused, dense_score,
    lexical_score, dense_rank, lexical_rank)``, best first, at most ``k`` long.

    Ranks are 1-based and reported as they were in the arm that produced them,
    so a caller can see that a result came third lexically and not at all
    densely.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if k < 1:
        raise ValueError("k must be positive")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    # A zero-weighted arm is dropped rather than multiplied by zero. Keeping it
    # would let its hits enter the output scoring 0.0 and pad the results up to
    # k — a caller who asked for dense-only would get lexical matches back,
    # ranked last, looking like weak dense hits. `Collection.search` already
    # skips running the unused arm; this makes `fuse` agree when handed both.
    use_dense = alpha > 0.0
    use_lexical = alpha < 1.0

    dense_ranks = (
        {chunk_id: index + 1 for index, (chunk_id, _) in enumerate(dense)} if use_dense else {}
    )
    lexical_ranks = (
        {chunk_id: index + 1 for index, (chunk_id, _) in enumerate(lexical)} if use_lexical else {}
    )
    dense_scores = dict(dense) if use_dense else {}
    lexical_scores = dict(lexical) if use_lexical else {}

    fused: dict[str, float] = {}
    for chunk_id, rank in dense_ranks.items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + alpha / (rrf_k + rank)
    for chunk_id, rank in lexical_ranks.items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + (1.0 - alpha) / (rrf_k + rank)

    # Ties broken by chunk_id so the order is stable across runs and platforms.
    # Dict iteration order would otherwise depend on insertion, which depends on
    # which arm ran first — reproducible until somebody reorders two lines.
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    return [
        (
            chunk_id,
            score,
            dense_scores.get(chunk_id),
            lexical_scores.get(chunk_id),
            dense_ranks.get(chunk_id),
            lexical_ranks.get(chunk_id),
        )
        for chunk_id, score in ordered[:k]
    ]


def normalise_filters(filters: Mapping[str, object] | None) -> dict[str, list[str]]:
    """Accept ``{"source_id": "a"}`` or ``{"source_id": ["a", "b"]}`` alike.

    Unknown keys raise. A filter that is silently ignored is the worst kind:
    the caller believes they restricted the search, the results look plausible,
    and nothing says otherwise — which for a product filtering to one user's
    documents is a disclosure bug rather than a quality one.
    """
    if not filters:
        return {}

    allowed = {"source_id", "source_type"}
    unknown = set(filters) - allowed
    if unknown:
        raise ValueError(
            f"unknown filter key(s): {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(allowed))}"
        )

    out: dict[str, list[str]] = {}
    for key, value in filters.items():
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, Sequence):
            values = [str(item) for item in value]
        else:
            values = [str(value)]
        if not values:
            raise ValueError(f"filter {key!r} is empty; omit it instead of passing nothing")
        out[key] = values
    return out
