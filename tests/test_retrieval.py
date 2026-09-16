"""Hybrid retrieval — the fusion, and the two places it goes wrong at the edges.

The fusion tests are arithmetic and run anywhere. The end-to-end tests use
``HashingEmbedder``, which knows nothing about meaning and matches on word
overlap alone; that is enough to prove *plumbing* — that both arms run, that
filters bite before the cut, that a chunk with no vector is still findable — and
it is deliberately not used to make any claim about retrieval quality, which
belongs to the model and is tested against the real one.

Two behaviours here are worth more than the rest:

* **A filter is applied before the top-k cut, not after.** Filtering afterwards
  returns three results for a filter matching hundreds, and looks like a
  relevance problem rather than a bug.
* **RRF is invariant to the BM25 scale.** That is the whole reason it was chosen
  over normalising the two arms onto a common range, so it is pinned here: if
  someone swaps in min-max normalisation, this test fails.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from retrieval_core import models
from retrieval_core.chunking import Source, chunk_spans
from retrieval_core.embedding import HashingEmbedder
from retrieval_core.provenance import FileLocator, SourceType
from retrieval_core.retrieval import RRF_K, SearchResult, fuse, normalise_filters
from retrieval_core.spans import Span
from retrieval_core.storage import Collection, IndexMismatchError

WHEN = datetime(2026, 9, 15, tzinfo=UTC)
DIM = 64

SPEC = models.NOMIC_V15_INT8
requires_weights = pytest.mark.skipif(
    not all((models.default_cache_dir() / SPEC.model_id / f.path).exists() for f in SPEC.files),
    reason="model weights not present; run NomicEmbedder.load(allow_download=True) once",
)


def source(source_id: str, title: str = "doc.md", kind: SourceType = SourceType.FILE) -> Source:
    return Source(
        source_id=source_id,
        source_type=kind,
        source_uri=f"C:/docs/{title}",
        source_title=title,
        content_hash=f"hash-{source_id}",
    )


def chunks_for(source_id: str, texts: list[str], kind: SourceType = SourceType.FILE):
    spans = []
    position = 0
    for text in texts:
        spans.append(Span(text=text, locator=FileLocator(position, position + len(text), page=1)))
        position += len(text)
    return chunk_spans(
        spans,
        source(source_id, kind=kind),
        strategy="steps",  # one chunk per span, so the test controls the count
        ingested_at=WHEN,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=DIM)


@pytest.fixture
def collection(tmp_path: Path, embedder: HashingEmbedder):
    with Collection.open(
        tmp_path / "coll", embedding_model=embedder.model_id, dimension=DIM
    ) as coll:
        yield coll


def fill(
    collection: Collection,
    embedder: HashingEmbedder,
    source_id: str,
    texts: list[str],
    kind: SourceType = SourceType.FILE,
) -> None:
    chunks = chunks_for(source_id, texts, kind=kind)
    collection.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))


class TestFuse:
    def test_computes_weighted_rrf(self) -> None:
        dense = [("a", 0.9), ("b", 0.8)]
        lexical = [("b", -1.0), ("c", -2.0)]

        fused = dict((row[0], row[1]) for row in fuse(dense, lexical, alpha=0.5, k=10))

        assert fused["a"] == pytest.approx(0.5 / (RRF_K + 1))
        assert fused["b"] == pytest.approx(0.5 / (RRF_K + 2) + 0.5 / (RRF_K + 1))
        assert fused["c"] == pytest.approx(0.5 / (RRF_K + 2))

    def test_agreement_beats_a_single_arms_confidence(self) -> None:
        # The property that makes fusion worth having: a chunk both arms rank
        # third beats one that a single arm ranked first. An arm can be
        # confidently wrong; two arms agreeing rarely are.
        dense = [("solo", 0.99), ("x", 0.9), ("agreed", 0.8)]
        lexical = [("y", -1.0), ("z", -2.0), ("agreed", -3.0)]

        order = [row[0] for row in fuse(dense, lexical, alpha=0.5, k=5)]

        assert order[0] == "agreed"

    def test_is_invariant_to_the_lexical_scale(self) -> None:
        # The reason RRF was chosen over normalising both arms onto a common
        # range. BM25 has no ceiling and its magnitude depends on the corpus;
        # multiplying every lexical score by 100 must change nothing, because
        # only the ordering carries information.
        dense = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        lexical = [("c", -1.2), ("a", -3.4), ("d", -9.9)]

        small = fuse(dense, lexical, alpha=0.5, k=4)
        large = fuse(dense, [(cid, score * 100) for cid, score in lexical], alpha=0.5, k=4)

        assert [row[0] for row in small] == [row[0] for row in large]
        assert [row[1] for row in small] == [row[1] for row in large]

    def test_is_invariant_to_the_dense_scale_too(self) -> None:
        dense = [("a", 0.9), ("b", 0.7)]
        lexical = [("b", -1.0)]

        base = [row[0] for row in fuse(dense, lexical, alpha=0.5, k=4)]
        shifted = [row[0] for row in fuse([(c, s * 0.01) for c, s in dense], lexical, k=4)]

        assert base == shifted

    def test_alpha_one_ignores_the_lexical_arm(self) -> None:
        fused = fuse([("a", 0.9)], [("b", -1.0)], alpha=1.0, k=5)

        assert [row[0] for row in fused] == ["a"]

    def test_alpha_zero_ignores_the_dense_arm(self) -> None:
        fused = fuse([("a", 0.9)], [("b", -1.0)], alpha=0.0, k=5)

        assert [row[0] for row in fused] == ["b"]

    def test_alpha_shifts_the_winner(self) -> None:
        dense = [("dense-pick", 0.9)]
        lexical = [("lexical-pick", -1.0)]

        assert fuse(dense, lexical, alpha=0.9, k=2)[0][0] == "dense-pick"
        assert fuse(dense, lexical, alpha=0.1, k=2)[0][0] == "lexical-pick"

    def test_carries_both_arms_scores_and_ranks(self) -> None:
        fused = fuse([("a", 0.9), ("b", 0.8)], [("b", -1.5)], alpha=0.5, k=5)
        rows = {row[0]: row for row in fused}

        assert rows["b"][2] == pytest.approx(0.8)
        assert rows["b"][3] == pytest.approx(-1.5)
        assert rows["b"][4] == 2  # second densely
        assert rows["b"][5] == 1  # first lexically
        # A None here is information: the lexical arm did not return "a".
        assert rows["a"][3] is None
        assert rows["a"][5] is None

    def test_truncates_to_k(self) -> None:
        dense = [(f"c{index}", 1.0 - index / 100) for index in range(50)]

        assert len(fuse(dense, [], alpha=1.0, k=7)) == 7

    def test_ties_break_deterministically(self) -> None:
        # Same rank in the same arm, so identical fused scores. Without an
        # explicit tiebreak the order would follow dict insertion, which follows
        # whichever arm was assembled first — reproducible until somebody
        # reorders two lines.
        first = fuse([("zeta", 0.5)], [("alpha", -1.0)], alpha=0.5, k=5)
        second = fuse([("zeta", 0.5)], [("alpha", -1.0)], alpha=0.5, k=5)

        assert [row[0] for row in first] == [row[0] for row in second] == ["alpha", "zeta"]

    def test_empty_arms_give_nothing(self) -> None:
        assert fuse([], [], alpha=0.5, k=5) == []

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_rejects_an_out_of_range_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must be"):
            fuse([], [], alpha=alpha)

    def test_rejects_a_useless_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            fuse([], [], k=0)

    def test_rejects_a_useless_rrf_k(self) -> None:
        with pytest.raises(ValueError, match="rrf_k must be positive"):
            fuse([], [], rrf_k=0)


class TestNormaliseFilters:
    def test_none_is_no_filter(self) -> None:
        assert normalise_filters(None) == {}
        assert normalise_filters({}) == {}

    def test_a_bare_string_becomes_a_list(self) -> None:
        assert normalise_filters({"source_id": "s1"}) == {"source_id": ["s1"]}

    def test_a_list_is_kept(self) -> None:
        assert normalise_filters({"source_id": ["s1", "s2"]}) == {"source_id": ["s1", "s2"]}

    def test_an_enum_survives(self) -> None:
        assert normalise_filters({"source_type": SourceType.RECORDING}) == {
            "source_type": ["recording"]
        }

    def test_an_unknown_key_raises(self) -> None:
        # A silently ignored filter is a disclosure bug, not a quality one: the
        # caller believes they restricted the search and nothing says otherwise.
        with pytest.raises(ValueError, match="unknown filter key"):
            normalise_filters({"souce_id": "s1"})

    def test_an_empty_value_raises(self) -> None:
        with pytest.raises(ValueError, match="omit it instead"):
            normalise_filters({"source_id": []})


class TestSearch:
    def test_finds_the_obvious_match(self, collection: Collection, embedder) -> None:
        fill(
            collection,
            embedder,
            "s1",
            [
                "the quarterly vat return is filed monthly",
                "sourdough proving basket and banneton",
                "petty cash reconciliation on fridays",
            ],
        )

        results = collection.search("quarterly vat return", k=1, embedder=embedder)

        assert len(results) == 1
        assert "vat return" in results[0].text

    def test_returns_search_results_that_can_cite(self, collection: Collection, embedder) -> None:
        fill(collection, embedder, "s1", ["the quarterly vat return"])

        result = collection.search("vat", k=1, embedder=embedder)[0]

        assert isinstance(result, SearchResult)
        assert result.cite() == "doc.md, p. 1"

    def test_an_empty_collection_returns_nothing(self, collection: Collection, embedder) -> None:
        assert collection.search("anything", embedder=embedder) == []

    def test_respects_k(self, collection: Collection, embedder) -> None:
        fill(
            collection,
            embedder,
            "s1",
            [f"chunk number {index} about ledgers" for index in range(20)],
        )

        assert len(collection.search("ledgers", k=3, embedder=embedder)) == 3

    def test_lexical_only_needs_no_embedder(self, collection: Collection, embedder) -> None:
        # §5's storage must work without the embedding extra installed. A
        # product that only wants BM25 must not be made to load a model.
        fill(collection, embedder, "s1", ["the quarterly vat return"])

        results = collection.search("vat", k=1, alpha=0.0)

        assert len(results) == 1
        assert results[0].dense_score is None

    def test_dense_search_without_a_vector_says_what_to_do(self, collection: Collection) -> None:
        with pytest.raises(ValueError, match="alpha=0.0 for lexical-only"):
            collection.search("anything", alpha=0.5)

    def test_accepts_a_precomputed_query_vector(self, collection: Collection, embedder) -> None:
        fill(collection, embedder, "s1", ["the quarterly vat return"])

        results = collection.search(
            "vat", k=1, alpha=1.0, query_vector=embedder.embed_query("vat return")
        )

        assert len(results) == 1

    def test_a_different_model_is_refused(self, collection: Collection, embedder) -> None:
        # §6 freezes the model per collection. Vectors from two models are not
        # comparable, and the failure is silent unless something checks.
        fill(collection, embedder, "s1", ["the quarterly vat return"])

        class Impostor(HashingEmbedder):
            @property
            def model_id(self) -> str:
                return "some-other-model"

        with pytest.raises(IndexMismatchError, match="reindex to change model"):
            collection.search("vat", embedder=Impostor(dimension=DIM))

    def test_a_wrong_dimension_is_refused(self, collection: Collection, embedder) -> None:
        fill(collection, embedder, "s1", ["the quarterly vat return"])

        with pytest.raises(ValueError, match="collection expects"):
            collection.search("vat", alpha=1.0, query_vector=np.zeros(DIM + 8, dtype=np.float32))

    @pytest.mark.parametrize("alpha", [-0.5, 2.0])
    def test_rejects_an_out_of_range_alpha(self, collection: Collection, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must be"):
            collection.search("x", alpha=alpha)

    def test_rejects_a_useless_k(self, collection: Collection) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            collection.search("x", k=0, alpha=0.0)


class TestBothArmsContribute:
    def test_lexical_finds_what_a_bag_of_words_vector_cannot(
        self, collection: Collection, embedder
    ) -> None:
        # A rare exact term is BM25's strength. The hashing embedder spreads it
        # across one bucket among sixty-four and loses it in the noise.
        fill(
            collection,
            embedder,
            "s1",
            ["the invoice reference is ZQ-4417 and payment follows in thirty days"]
            + [f"unrelated paragraph {index} about scheduling and rotas" for index in range(30)],
        )

        lexical = collection.search("ZQ-4417", k=1, alpha=0.0)

        assert "ZQ-4417" in lexical[0].text

    def test_hybrid_keeps_what_each_arm_found(self, collection: Collection, embedder) -> None:
        fill(
            collection,
            embedder,
            "s1",
            [
                "the invoice reference is ZQ-4417",
                "invoice payment terms are thirty days",
            ],
        )

        results = collection.search("invoice ZQ-4417", k=2, alpha=0.5, embedder=embedder)

        assert len(results) == 2
        assert any(result.found_by_both for result in results)

    def test_a_chunk_with_no_vector_is_still_findable(
        self, collection: Collection, embedder
    ) -> None:
        # Upserting without vectors is allowed — a product may index text first
        # and embed later. Those chunks must not vanish from search entirely.
        collection.upsert(chunks_for("s1", ["a chunk that was never embedded"]))

        assert collection.search("never embedded", k=1, alpha=0.0)
        assert collection.search_dense(embedder.embed_query("never embedded"), k=5) == []


class TestFiltersBiteBeforeTheCut:
    """The bug this class exists for: filtering *after* the top-k cut.

    It looks like a relevance problem — the search returns two results for a
    filter matching a hundred chunks — and it survives casual testing, because
    with a small fixture the wanted source is usually in the global top-k anyway.
    """

    def build(self, collection: Collection, embedder) -> None:
        # Thirty chunks in s1 that all match the query well, and five in s2 that
        # match it less well. A post-hoc filter for s2 would return nothing.
        fill(collection, embedder, "s1", [f"ledger reconciliation note {i}" for i in range(30)])
        fill(collection, embedder, "s2", [f"ledger appendix {i}" for i in range(5)])

    def test_a_source_filter_returns_the_filtered_top_k(
        self, collection: Collection, embedder
    ) -> None:
        self.build(collection, embedder)

        results = collection.search("ledger", k=5, filters={"source_id": "s2"}, embedder=embedder)

        assert len(results) == 5
        assert {result.provenance.source_id for result in results} == {"s2"}

    def test_it_holds_for_each_arm_alone(self, collection: Collection, embedder) -> None:
        self.build(collection, embedder)

        for alpha in (0.0, 1.0):
            results = collection.search(
                "ledger", k=5, filters={"source_id": "s2"}, alpha=alpha, embedder=embedder
            )
            assert {result.provenance.source_id for result in results} == {"s2"}, alpha

    def test_several_sources_at_once(self, collection: Collection, embedder) -> None:
        self.build(collection, embedder)

        results = collection.search(
            "ledger", k=35, filters={"source_id": ["s1", "s2"]}, embedder=embedder
        )

        assert {result.provenance.source_id for result in results} == {"s1", "s2"}

    def test_filtering_by_source_type(self, collection: Collection, embedder) -> None:
        fill(collection, embedder, "s1", ["ledger note from a document"])
        fill(collection, embedder, "s2", ["ledger note from a meeting"], kind=SourceType.RECORDING)

        results = collection.search(
            "ledger note", k=5, filters={"source_type": SourceType.RECORDING}, embedder=embedder
        )

        assert len(results) == 1
        assert results[0].provenance.source_type is SourceType.RECORDING

    def test_a_filter_matching_nothing_returns_nothing(
        self, collection: Collection, embedder
    ) -> None:
        self.build(collection, embedder)

        assert collection.search("ledger", filters={"source_id": "absent"}, embedder=embedder) == []

    def test_an_unknown_filter_key_raises(self, collection: Collection, embedder) -> None:
        with pytest.raises(ValueError, match="unknown filter key"):
            collection.search("ledger", filters={"surce_id": "s1"}, embedder=embedder)


class TestTheLexicalArmMatchesPartially:
    """Found while testing hybrid search: the lexical arm was returning nothing.

    `_fts_query` quoted each term and joined them with spaces, and FTS5's
    implicit operator is AND — so every term had to appear in the same chunk.
    A question ("When is the VAT return due?") matched nothing at all, because
    no chunk contains "when" and "is" and "due" together, and hybrid search
    quietly ran on one arm against exactly the queries people type.

    It was invisible from the tests that existed because they all searched for
    single words or for phrases that happened to appear verbatim.
    """

    def test_a_multi_word_query_matches_a_partial_hit(
        self, collection: Collection, embedder
    ) -> None:
        fill(collection, embedder, "s1", ["the quarterly vat return is filed"])

        assert collection.search_lexical("when is the vat return due", k=5)

    def test_more_matched_terms_outranks_fewer(self, collection: Collection, embedder) -> None:
        # What OR buys: BM25 ranks by how many terms matched and how rare they
        # are, which is the contribution fusion wants from this arm.
        fill(
            collection,
            embedder,
            "s1",
            [
                "quarterly vat return filing deadline",
                "quarterly report",
                "the sourdough proves overnight",
            ],
        )

        ranked = collection.search_lexical("quarterly vat return deadline", k=3)

        assert collection.get(ranked[0][0])[0] == "quarterly vat return filing deadline"

    def test_user_text_is_never_read_as_fts5_syntax(self, collection: Collection, embedder) -> None:
        # The safety half of the same function. These must be searched for as
        # words, and must not raise.
        fill(collection, embedder, "s1", ["profit and loss margin analysis"])

        for query in [
            "profit AND loss",
            "margin*",
            "NEAR(profit loss)",
            'unbalanced " quote',
            '""',
        ]:
            collection.search_lexical(query, k=5)

        assert collection.search_lexical("profit AND loss", k=5)

    def test_a_query_of_only_punctuation_finds_nothing(
        self, collection: Collection, embedder
    ) -> None:
        fill(collection, embedder, "s1", ["profit and loss"])

        assert collection.search_lexical("!?-", k=5) == []


class TestDeletionAndSearch:
    def test_a_deleted_source_is_not_returned(self, collection: Collection, embedder) -> None:
        fill(collection, embedder, "s1", ["the quarterly vat return"])
        fill(collection, embedder, "s2", ["another vat note entirely"])

        collection.delete_by_source("s1")
        results = collection.search("vat", k=10, embedder=embedder)

        assert {result.provenance.source_id for result in results} == {"s2"}

    def test_search_survives_the_vector_compaction(self, collection: Collection, embedder) -> None:
        # Deletion rewrites vector_row values. If search read stale rows it
        # would pair chunks with other chunks' vectors — every result plausible
        # and wrong.
        fill(collection, embedder, "s1", [f"first source note {i}" for i in range(5)])
        fill(collection, embedder, "s2", ["second source unmistakable zarquon note"])
        collection.delete_by_source("s1")

        results = collection.search_dense(embedder.embed_query("zarquon"), k=1)

        assert collection.get(results[0][0])[0] == "second source unmistakable zarquon note"


class TestExpand:
    def parent_and_children(self, collection: Collection, embedder):
        """A coarse pass and a fine pass over the same source.

        The two passes must differ in `content_hash`, because a chunk id is
        derived from source, hash and position — index the same source twice at
        the same position and the second upsert *replaces* the first. That is
        correct for re-ingest and a trap for small-to-big: a parent written with
        the child's own identity ends up pointing at itself.
        """
        from retrieval_core.chunking import ChunkingConfig

        coarse = Source(
            source_id="s1",
            source_type=SourceType.FILE,
            source_uri="C:/docs/doc.md",
            source_title="doc.md",
            content_hash="hash-s1-sections",
        )
        parent = chunk_spans(
            [
                Span(
                    text="the whole section, several paragraphs long",
                    locator=FileLocator(0, 42, page=1),
                )
            ],
            coarse,
            strategy="steps",
            ingested_at=WHEN,
        )[0]
        collection.upsert([parent])

        children = chunk_spans(
            [Span(text="one precise sentence", locator=FileLocator(0, 20, page=1))],
            source("s1"),
            strategy="steps",
            config=ChunkingConfig(parent_id=parent.chunk_id),
            ingested_at=WHEN,
        )
        collection.upsert(children, embedder.embed_documents([c.text for c in children]))
        return parent, children

    def test_returns_the_parents_text(self, collection: Collection, embedder) -> None:
        self.parent_and_children(collection, embedder)

        result = collection.search("precise sentence", k=1, alpha=0.0)[0]

        assert result.text == "one precise sentence"
        assert collection.expand(result) == "the whole section, several paragraphs long"

    def test_the_parent_is_not_returned_as_its_own_search_hit(
        self, collection: Collection, embedder
    ) -> None:
        # The parent is indexed for expansion, not for matching — it has no
        # vector, so it can still surface lexically, and that is fine. What
        # must hold is that expanding the child reaches it.
        _, children = self.parent_and_children(collection, embedder)
        result = collection.search("precise sentence", k=1, alpha=0.0)[0]

        assert result.chunk_id == children[0].chunk_id

    def test_a_self_referential_parent_expands_to_itself(
        self, collection: Collection, embedder
    ) -> None:
        # What the id collision above produces if a product gets it wrong.
        # Following the link would cost a query and return the same text.
        from retrieval_core.chunking import ChunkingConfig

        first = chunks_for("s1", ["a chunk that claims to be its own parent"])[0]
        again = chunk_spans(
            [
                Span(
                    text="a chunk that claims to be its own parent",
                    locator=FileLocator(0, 40, page=1),
                )
            ],
            source("s1"),
            strategy="steps",
            config=ChunkingConfig(parent_id=first.chunk_id),
            ingested_at=WHEN,
        )
        collection.upsert(again, embedder.embed_documents([c.text for c in again]))

        result = collection.search("own parent", k=1, alpha=0.0)[0]

        assert result.chunk_id == result.provenance.parent_id
        assert collection.expand(result) == "a chunk that claims to be its own parent"

    def test_a_chunk_with_no_parent_expands_to_itself(
        self, collection: Collection, embedder
    ) -> None:
        fill(collection, embedder, "s1", ["a standalone chunk"])

        result = collection.search("standalone", k=1, alpha=0.0)[0]

        assert collection.expand(result) == "a standalone chunk"

    def test_a_missing_parent_does_not_invent_context(
        self, collection: Collection, embedder
    ) -> None:
        # A product may index children without their parents. Returning the
        # child's own text is right; returning anything else would put text in
        # front of generation that no citation covers.
        from retrieval_core.chunking import ChunkingConfig

        children = chunk_spans(
            [Span(text="an orphaned child chunk", locator=FileLocator(0, 23, page=1))],
            source("s1"),
            strategy="steps",
            config=ChunkingConfig(parent_id="a-parent-that-was-never-indexed"),
            ingested_at=WHEN,
        )
        collection.upsert(children, embedder.embed_documents([c.text for c in children]))

        result = collection.search("orphaned", k=1, alpha=0.0)[0]

        assert collection.expand(result) == "an orphaned child chunk"


@pytest.fixture(scope="module")
def real():
    from retrieval_core.embedding import NomicEmbedder

    return NomicEmbedder.load()


CORPUS = [
    # The answer, worded nothing like the query below: no "VAT", no "deadline".
    "Value added tax must be paid to the revenue one month and seven days "
    "after the end of the accounting period.",
    # The decoy: says "VAT" twice and answers a different question.
    "The VAT return form asks for the box 6 total excluding VAT.",
    "Petty cash is counted every Friday afternoon by two people.",
    "The sourdough proves overnight in a banneton at room temperature.",
    "Employee expense claims are reimbursed in the following month's payroll.",
]
ANSWER, DECOY = CORPUS[0], CORPUS[1]


@pytest.fixture
def indexed(tmp_path: Path, real):
    with Collection.open(
        tmp_path / "coll", embedding_model=real.model_id, dimension=real.dimension
    ) as coll:
        chunks = chunks_for("s1", CORPUS)
        coll.upsert(chunks, real.embed_documents([chunk.text for chunk in chunks]))
        yield coll


@requires_weights
class TestHybridWithTheRealModel:
    """The claim the increment makes, against the model rather than the stub.

    `HashingEmbedder` cannot show this: it matches on word overlap, so its dense
    arm agrees with BM25 by construction and fusion has nothing to add. The
    point of a dense arm is the case where the words differ and the meaning does
    not, and only real weights produce that.

    The query is "VAT payment deadline". The answer contains none of those three
    words — it says "value added tax", "paid", and "one month and seven days
    after" — so BM25 cannot reach it and settles for the chunk that repeats
    "VAT". Measured cosines: answer 0.696, decoy 0.635.
    """

    QUERY = "VAT payment deadline"

    def test_dense_finds_the_paraphrase_that_lexical_misses(self, indexed, real) -> None:
        dense = indexed.search(self.QUERY, k=1, alpha=1.0, embedder=real)
        lexical = indexed.search(self.QUERY, k=1, alpha=0.0)

        assert dense[0].text == ANSWER
        assert lexical[0].text == DECOY

    def test_hybrid_returns_both_readings(self, indexed, real) -> None:
        results = indexed.search(self.QUERY, k=2, alpha=0.5, embedder=real)

        assert {result.text for result in results} == {ANSWER, DECOY}

    def test_alpha_is_a_lean_not_a_switch(self, indexed, real) -> None:
        """Agreement across arms is hard to override, and that is deliberate.

        The decoy is second densely *and* first lexically; the answer is first
        densely and absent lexically. Working the RRF arithmetic through with
        rrf_k=60, the answer only takes the top slot above alpha 0.984 —
        anywhere below that, being well-placed in both arms wins.

        So alpha tilts the ranking rather than handing it over. A product that
        wants one arm alone has to say so with 0.0 or 1.0, which is the honest
        interface: "mostly dense" is not the same request as "dense only", and a
        weight that silently became a switch at 0.9 would hide that.
        """
        assert indexed.search(self.QUERY, k=1, alpha=1.0, embedder=real)[0].text == ANSWER
        for alpha in (0.9, 0.5, 0.1):
            top = indexed.search(self.QUERY, k=1, alpha=alpha, embedder=real)[0]
            assert top.text == DECOY, alpha
            assert top.found_by_both

    def test_the_dense_score_is_a_real_cosine(self, indexed, real) -> None:
        # Unlike the fused score this one is comparable between queries, which
        # is why fusion carries it through instead of discarding it.
        on_topic = indexed.search(self.QUERY, k=1, alpha=1.0, embedder=real)[0]
        off_topic = indexed.search(
            "quantum chromodynamics lattice gauge", k=1, alpha=1.0, embedder=real
        )[0]

        assert -1.0 <= off_topic.dense_score <= 1.0
        assert on_topic.dense_score > off_topic.dense_score + 0.1

    def test_an_irrelevant_query_still_scores_lower(self, indexed, real) -> None:
        """What min-max normalisation would destroy.

        Normalising each result set onto [0, 1] makes the top hit exactly 1.0
        for every query, so a search that found nothing looks identical to one
        that found the answer. On the raw scale it does not: measured, the best
        hit for a nonsense query is 0.511 against 0.696 for a real one.

        Note the floor is around 0.35-0.55 rather than 0 — this model puts
        unrelated English text well above zero, so a product choosing a
        relevance cutoff has to calibrate it rather than assume 0.5 means
        "somewhat related".
        """
        on_topic = indexed.search(self.QUERY, k=1, alpha=1.0, embedder=real)[0]
        off_topic = indexed.search(
            "quantum chromodynamics lattice gauge", k=1, alpha=1.0, embedder=real
        )[0]

        assert off_topic.dense_score < 0.60
        assert on_topic.dense_score > 0.65
