"""Embedding, where being wrong looks exactly like being right.

A broken chunker throws. A broken storage layer throws. A broken embedder
returns a perfectly well-formed array of 768 floats and the index built from it
looks completely normal — retrieval is just worse, for the life of the
collection, with nothing anywhere to say so. Every test here exists because of
some way that can happen:

* the task prefixes silently not applied
* the Matryoshka truncation done after the L2 normalisation instead of before
* padding averaged into the mean, so a vector depends on its batch neighbours
* the INT8 export's batch sensitivity, which is real and measured here

The maths tests run everywhere. The tests that need 137 MB of weights skip when
the weights are absent, which is the normal state in CI — so the model tests are
a local gate, and the file says so rather than pretending CI covers them.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from retrieval_core import models
from retrieval_core.embedding import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    Embedder,
    HashingEmbedder,
    NomicEmbedder,
    pool_and_normalize,
)

SPEC = models.NOMIC_V15_INT8


def weights_present() -> bool:
    """Cheap presence check — size only, no 137 MB hash at collection time."""
    cache = models.default_cache_dir() / SPEC.model_id
    return all((cache / entry.path).exists() for entry in SPEC.files)


requires_weights = pytest.mark.skipif(
    not weights_present(),
    reason="model weights not present; run NomicEmbedder.load(allow_download=True) once",
)


@pytest.fixture(scope="module")
def embedder() -> NomicEmbedder:
    return NomicEmbedder.load()


class TestPoolAndNormalize:
    def test_returns_unit_vectors(self) -> None:
        hidden = np.random.default_rng(0).normal(size=(4, 6, 12))
        mask = np.ones((4, 6), dtype=np.int64)

        vectors = pool_and_normalize(hidden, mask)

        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_truncated_vectors_are_also_unit_vectors(self) -> None:
        # This is the whole Matryoshka correctness claim in one assertion. The
        # truncation happens before the normalisation, so the short vector is a
        # unit vector in its own right and a dot product against it is still a
        # cosine.
        hidden = np.random.default_rng(1).normal(size=(5, 7, 16))
        mask = np.ones((5, 7), dtype=np.int64)

        vectors = pool_and_normalize(hidden, mask, dimension=4)

        assert vectors.shape == (5, 4)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_truncating_after_normalising_would_not_survive_this(self) -> None:
        # The mutation the test above is aimed at: swap the last two steps and
        # the result is the leading slice of a full-width unit vector, whose
        # length is whatever fraction of the mass happened to land there. If
        # that slice were accidentally unit length the test above would pass on
        # a broken implementation, so check here that it is not.
        hidden = np.random.default_rng(2).normal(size=(5, 7, 16))
        mask = np.ones((5, 7), dtype=np.int64)

        full = pool_and_normalize(hidden, mask)
        wrong = full[:, :4]

        assert not np.allclose(np.linalg.norm(wrong, axis=1), 1.0)
        assert not np.allclose(wrong, pool_and_normalize(hidden, mask, dimension=4))

    def test_truncation_keeps_the_leading_dimensions(self) -> None:
        # Matryoshka only works because the *first* k dimensions carry the
        # signal. Slicing from anywhere else would still produce unit vectors.
        hidden = np.random.default_rng(3).normal(size=(3, 5, 10))
        mask = np.ones((3, 5), dtype=np.int64)

        short = pool_and_normalize(hidden, mask, dimension=6)
        full = pool_and_normalize(hidden, mask)

        # Same direction in the leading subspace, only rescaled.
        scaled = full[:, :6] / np.linalg.norm(full[:, :6], axis=1, keepdims=True)
        assert np.allclose(short, scaled, atol=1e-6)

    def test_padding_is_excluded_from_the_mean(self) -> None:
        # The same text embedded alone and embedded beside a longer one must
        # give the same vector. If padding were averaged in as zeros, a vector
        # would depend on its batch neighbours — reproducible only by accident.
        rng = np.random.default_rng(4)
        real = rng.normal(size=(1, 3, 8))
        padded = np.concatenate([real, rng.normal(size=(1, 5, 8))], axis=1)

        alone = pool_and_normalize(real, np.ones((1, 3), dtype=np.int64))
        in_batch = pool_and_normalize(padded, np.array([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64))

        assert np.allclose(alone, in_batch, atol=1e-6)

    def test_padding_values_cannot_leak(self) -> None:
        rng = np.random.default_rng(5)
        real = rng.normal(size=(1, 3, 8))
        mask = np.array([[1, 1, 1, 0]], dtype=np.int64)

        noisy = rng.normal(size=(1, 1, 8)) * 1e6
        quiet = pool_and_normalize(np.concatenate([real, np.zeros((1, 1, 8))], axis=1), mask)
        loud = pool_and_normalize(np.concatenate([real, noisy], axis=1), mask)

        assert np.allclose(quiet, loud, atol=1e-6)

    def test_an_all_padding_row_is_zero_not_nan(self) -> None:
        # Division by a zero token count. A NaN here would propagate into the
        # vector file and poison every search that touched it.
        hidden = np.random.default_rng(6).normal(size=(2, 4, 8))
        mask = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int64)

        vectors = pool_and_normalize(hidden, mask)

        assert np.isfinite(vectors).all()
        assert np.allclose(vectors[1], 0.0)

    def test_a_constant_row_normalises_to_zero_not_nan(self) -> None:
        # Zero variance into the layer norm. Rare, but a degenerate chunk of
        # repeated tokens can get close.
        hidden = np.full((1, 3, 8), 0.5)
        mask = np.ones((1, 3), dtype=np.int64)

        vectors = pool_and_normalize(hidden, mask)

        assert np.isfinite(vectors).all()

    def test_output_is_float32_to_match_the_vector_file(self) -> None:
        hidden = np.random.default_rng(7).normal(size=(2, 3, 8))
        vectors = pool_and_normalize(hidden, np.ones((2, 3), dtype=np.int64))

        assert vectors.dtype == np.float32

    def test_rejects_a_mismatched_mask(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            pool_and_normalize(np.zeros((2, 3, 8)), np.ones((2, 4), dtype=np.int64))

    def test_rejects_wrongly_shaped_hidden_states(self) -> None:
        with pytest.raises(ValueError, match="batch, sequence, dim"):
            pool_and_normalize(np.zeros((3, 8)), np.ones((3,), dtype=np.int64))

    @pytest.mark.parametrize("dimension", [0, -1, 9])
    def test_rejects_an_impossible_dimension(self, dimension: int) -> None:
        with pytest.raises(ValueError, match="dimension must be"):
            pool_and_normalize(np.zeros((1, 2, 8)), np.ones((1, 2), dtype=np.int64), dimension)


class TestPrefixDiscipline:
    def test_the_prefixes_are_what_nomic_expects(self) -> None:
        # A typo here is not a crash, it is a permanent quality tax on every
        # collection built with the build that carries it.
        assert DOCUMENT_PREFIX == "search_document: "
        assert QUERY_PREFIX == "search_query: "

    def test_no_public_method_accepts_a_prefix(self) -> None:
        # §6: "the library applies these internally — never left to callers,
        # never a configuration option". A `prefix=` or `is_query=` parameter is
        # how that rule gets broken, because the default is right until one
        # caller passes the other value.
        for method in (NomicEmbedder.embed_documents, NomicEmbedder.embed_query):
            parameters = set(inspect.signature(method).parameters)
            assert parameters <= {"self", "texts", "text"}, parameters

    def test_document_and_query_are_separate_methods(self) -> None:
        assert hasattr(NomicEmbedder, "embed_documents")
        assert hasattr(NomicEmbedder, "embed_query")


class TestHashingEmbedder:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(HashingEmbedder(), Embedder)

    def test_is_deterministic_across_instances(self) -> None:
        # Two processes indexing the same text must agree, or an incrementally
        # updated collection ends up with two neighbourhoods for one document.
        first = HashingEmbedder(dimension=32).embed_documents(["the same words"])
        second = HashingEmbedder(dimension=32).embed_documents(["the same words"])

        assert np.array_equal(first, second)

    def test_produces_unit_vectors(self) -> None:
        vectors = HashingEmbedder(dimension=32).embed_documents(["alpha beta", "gamma"])

        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_tracks_word_overlap(self) -> None:
        embedder = HashingEmbedder(dimension=256)
        vectors = embedder.embed_documents(
            ["invoice total due date", "invoice total due amount", "sourdough proving basket"]
        )

        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_query_returns_one_dimension(self) -> None:
        assert HashingEmbedder().embed_query("anything").shape == (64,)

    def test_empty_input_keeps_its_shape(self) -> None:
        # An empty batch has to return (0, dim), not (0,), or vstack into the
        # vector file fails on whichever caller first has nothing to add.
        assert HashingEmbedder(dimension=16).embed_documents([]).shape == (0, 16)

    def test_empty_text_is_zero_not_nan(self) -> None:
        assert np.allclose(HashingEmbedder().embed_query(""), 0.0)

    def test_model_id_marks_it_as_not_a_real_model(self) -> None:
        # The safeguard against this ever backing a real collection: §6's frozen
        # model check refuses to open one embedded with a different model_id.
        assert HashingEmbedder().model_id == "hashing-test-v1"
        assert HashingEmbedder().model_id != SPEC.model_id

    def test_rejects_an_impossible_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            HashingEmbedder(dimension=0)


@requires_weights
class TestNomicEmbedder:
    def test_satisfies_the_protocol(self, embedder: NomicEmbedder) -> None:
        assert isinstance(embedder, Embedder)

    def test_reports_the_pinned_identity(self, embedder: NomicEmbedder) -> None:
        assert embedder.model_id == SPEC.model_id
        assert embedder.dimension == 768

    def test_produces_unit_vectors(self, embedder: NomicEmbedder) -> None:
        vectors = embedder.embed_documents(["one", "two", "three"])

        assert vectors.shape == (3, 768)
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_the_prefixes_actually_reach_the_model(self, embedder: NomicEmbedder) -> None:
        # Identical text, embedded once as a document and once as a query. If
        # the prefixes were dropped these would be the same vector. They are
        # not, and the gap is the whole reason §6 makes them mandatory.
        text = "The quarterly VAT return must be filed within one month of the period end."

        similarity = float(embedder.embed_documents([text])[0] @ embedder.embed_query(text))

        assert similarity < 0.99

    def test_ranks_a_related_document_above_an_unrelated_one(self, embedder: NomicEmbedder) -> None:
        documents = [
            "Late filing of a VAT return attracts a penalty point under the new regime.",
            "Bake the sourdough at 230C for twenty minutes with steam.",
        ]
        scores = embedder.embed_documents(documents) @ embedder.embed_query(
            "When is the VAT return due?"
        )

        assert scores[0] > scores[1]

    def test_a_vector_does_not_depend_on_its_neighbours(self, embedder: NomicEmbedder) -> None:
        # The default is one forward pass per text, so this is exact equality
        # rather than a tolerance: the same text embedded alone and embedded
        # beside a much longer one must be the same bytes.
        text = "A short line."
        alone = embedder.embed_documents([text])[0]
        beside = embedder.embed_documents([text, "A far longer line. " * 40])[0]

        assert np.array_equal(alone, beside)

    def test_reindexing_reproduces_the_same_vectors(self, embedder: NomicEmbedder) -> None:
        # Products index incrementally. A chunk re-embedded after an edit
        # elsewhere in the document has to land where it landed before, or the
        # collection slowly fills with two neighbourhoods for the same text.
        texts = ["first paragraph", "second paragraph", "third paragraph"]

        assert np.array_equal(embedder.embed_documents(texts), embedder.embed_documents(texts))

    def test_the_int8_export_is_still_batch_sensitive(self, embedder: NomicEmbedder) -> None:
        """Characterisation, and a canary.

        Batching different texts through this export changes their vectors,
        because dynamic quantisation derives activation scales from each
        tensor's range at runtime. That measurement is why ``batch_size``
        defaults to 1 and costs us roughly half the throughput.

        If a future export fixes it, this test fails — and that failure is the
        signal to raise the default and take the speed back.
        """
        texts = [
            f"Paragraph {index}: a distinct sentence about topic {index}." for index in range(8)
        ]
        batched = NomicEmbedder.load(batch_size=8).embed_documents(texts)
        one_at_a_time = embedder.embed_documents(texts)

        similarity = (batched * one_at_a_time).sum(axis=1)
        assert similarity.min() < 0.999, "batching no longer perturbs vectors; revisit batch_size=1"

    def test_batching_is_opt_in_and_still_produces_valid_vectors(self) -> None:
        # Raising it is allowed — it is a documented trade, not a trap.
        batched = NomicEmbedder.load(batch_size=4)
        vectors = batched.embed_documents([f"line number {index}" for index in range(9)])

        assert vectors.shape == (9, 768)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_empty_input_keeps_its_shape(self, embedder: NomicEmbedder) -> None:
        assert embedder.embed_documents([]).shape == (0, 768)

    def test_truncated_dimension_still_ranks_correctly(self) -> None:
        # §6 keeps 256 available for the day RAM becomes a complaint. It is only
        # available if the ordering survives the truncation.
        short = NomicEmbedder.load(dimension=256)
        documents = [
            "Late filing of a VAT return attracts a penalty point under the new regime.",
            "Bake the sourdough at 230C for twenty minutes with steam.",
        ]
        vectors = short.embed_documents(documents)
        scores = vectors @ short.embed_query("When is the VAT return due?")

        assert vectors.shape == (2, 256)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        assert scores[0] > scores[1]

    def test_rejects_a_dimension_wider_than_the_model(self) -> None:
        with pytest.raises(ValueError, match="dimension must be"):
            NomicEmbedder.load(dimension=1024)

    def test_rejects_an_impossible_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            NomicEmbedder.load(batch_size=0)

    def test_overlong_text_is_truncated_not_rejected(self, embedder: NomicEmbedder) -> None:
        # A product will eventually hand in something oversized. Failing the
        # whole batch would lose the other chunks with it.
        vector = embedder.embed_documents(["padding. " * 4000])[0]

        assert np.isfinite(vector).all()


@requires_weights
class TestTokenizerCounter:
    def test_reserves_room_for_the_prefix(self, embedder: NomicEmbedder) -> None:
        # The caller's text is never what the model sees. Sizing a chunk against
        # the raw ceiling would have the tokenizer drop its tail, silently.
        assert embedder.max_context < SPEC.max_context

    def test_counts_real_tokens_not_words(self, embedder: NomicEmbedder) -> None:
        counter = embedder.token_counter

        assert counter.count("") == 0
        assert counter.count("hello") == 1
        # Subword splitting: a long unusual word is more than one token.
        assert counter.count("antidisestablishmentarianism") > 1

    def test_satisfies_the_token_counter_protocol(self, embedder: NomicEmbedder) -> None:
        from retrieval_core.tokens import TokenCounter

        assert isinstance(embedder.token_counter, TokenCounter)

    def test_is_tighter_than_the_estimate_it_replaces(self, embedder: NomicEmbedder) -> None:
        from retrieval_core.tokens import EstimatingCounter

        text = "The quarterly VAT return must be filed within one month of the period end."
        estimated = EstimatingCounter().count(text)
        actual = embedder.token_counter.count(text)

        # The estimate is documented as rounding up on purpose. This pins that
        # it does, rather than under-counting into a silent truncation.
        assert estimated >= actual


@requires_weights
class TestChunkingWithTheRealTokenizer:
    """The point of ``TokenizerCounter``: chunks sized against what the model reads.

    ``chunk_spans`` already takes a counter, so this is not new plumbing — it is
    a check that the two halves meet, and that a chunk sized by the real
    tokenizer actually fits through the real model without being truncated.
    """

    def test_chunks_sized_by_the_tokenizer_are_never_truncated(
        self, embedder: NomicEmbedder
    ) -> None:
        from datetime import UTC, datetime

        from retrieval_core.chunking import ChunkingConfig, Source, chunk_spans
        from retrieval_core.provenance import FileLocator, SourceType
        from retrieval_core.spans import Span, SpanKind

        source = Source(
            source_id="s1",
            source_type=SourceType.FILE,
            source_uri="C:/docs/handbook.md",
            source_title="handbook.md",
            content_hash="hash-s1",
        )
        paragraph = (
            "The reconciliation is performed monthly and signed off by the "
            "financial controller before the management pack is circulated. "
        )
        body = paragraph * 3
        spans = [
            Span(
                text=body,
                locator=FileLocator(char_start=index * len(body), char_end=(index + 1) * len(body)),
                kind=SpanKind.PROSE,
            )
            for index in range(12)
        ]

        chunks = chunk_spans(
            spans,
            source,
            strategy="prose",
            config=ChunkingConfig(prose_target=200),
            counter=embedder.token_counter,
            ingested_at=datetime(2026, 9, 15, tzinfo=UTC),
        )

        assert chunks
        for chunk in chunks:
            # What the model will actually be handed, prefix and specials
            # included. Over the ceiling here means the tokenizer drops the tail
            # and the index claims text it never embedded.
            budget = embedder.token_counter.count(chunk.text) + (
                SPEC.max_context - embedder.max_context
            )
            assert budget <= SPEC.max_context

    def test_the_chunks_embed(self, embedder: NomicEmbedder) -> None:
        from datetime import UTC, datetime

        from retrieval_core.chunking import Source, chunk_spans
        from retrieval_core.provenance import FileLocator, SourceType
        from retrieval_core.spans import Span, SpanKind

        source = Source(
            source_id="s2",
            source_type=SourceType.FILE,
            source_uri="C:/docs/notes.md",
            source_title="notes.md",
            content_hash="hash-s2",
        )
        spans = [
            Span(
                text="Petty cash is counted every Friday afternoon.",
                locator=FileLocator(char_start=0, char_end=44),
                kind=SpanKind.PROSE,
            ),
            Span(
                text="The float is five hundred pounds.",
                locator=FileLocator(char_start=45, char_end=78),
                kind=SpanKind.PROSE,
            ),
        ]
        chunks = chunk_spans(
            spans,
            source,
            counter=embedder.token_counter,
            ingested_at=datetime(2026, 9, 15, tzinfo=UTC),
        )

        vectors = embedder.embed_documents([chunk.text for chunk in chunks])

        assert vectors.shape == (len(chunks), 768)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
