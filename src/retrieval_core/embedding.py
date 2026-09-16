"""Turning text into vectors, and the three ways that goes silently wrong.

**One: the task prefixes are not optional and not yours.** Nomic was trained
with `search_document:` on indexed text and `search_query:` on queries, and
omitting them costs real retrieval quality. §6 settles the design consequence —
"the library applies these internally, never left to callers, never a
configuration option" — so this module offers ``embed_documents`` and
``embed_query`` as *separate methods* rather than one call with a flag. There is
no way to spell "embed this without a prefix", and no way to spell "embed this
document as a query", because both are bugs and a flag is how they happen: a
default argument flips somewhere up the stack and nothing fails, nothing logs,
retrieval just gets quietly worse.

**Two: the post-processing order is load-bearing.** The pipeline is mean-pool,
layer-norm, *truncate*, then L2-normalise — and the truncation sits before the
normalisation, not after. Matryoshka embeddings work because the first 256 of
768 dimensions carry most of the signal, but a unit vector's first 256
components are not themselves a unit vector. Truncating after normalising gives
short vectors of varying length, which makes a dot product no longer a cosine,
which quietly reorders results by how much of each vector's mass happened to
fall in the leading dimensions. The order here matches nomic's reference
implementation; ``pool_and_normalize`` is public and tested for exactly that.

**Three: the INT8 export does not batch cleanly, so by default it does not
batch.** This was measured here, not assumed. Feeding two different texts through
one forward pass changes *both* of their hidden states by around 10% against
running them separately; feeding the same text twice changes nothing at all,
bit for bit. That signature is dynamic quantisation — the INT8 graph computes
activation scales at runtime from each tensor's observed range, so a neighbour
in the batch moves the range and therefore moves your own numbers. After pooling
and normalisation it survives as roughly 0.985 cosine between a text embedded
alone and the same text embedded beside others.

0.985 sounds harmless and is not. §6 chose this model over BGE Small on the
evidence that same-domain similarity fell from 0.93–0.97 to 0.46–0.70, which is
what made vector dedup usable; a 0.985 noise floor that depends on batch
composition lands above that entire band and takes dedup back out. It also
breaks a quieter promise: re-indexing a corpus, or re-embedding one edited chunk,
would produce vectors that no longer match their neighbours from the first pass.

So ``batch_size`` defaults to 1 and each text gets its own forward pass.
Measured cost on this machine: 71 texts/s against 135 at batch 16 — under two
minutes per 10,000 chunks either way, for a one-off indexing job. Callers who
have measured their own tolerance can raise it.

The backend needs `onnxruntime` and `tokenizers`, which are the ``embedding``
extra rather than base dependencies. ``pool_and_normalize`` and
``HashingEmbedder`` need neither, so the tests that pin the maths above run
everywhere, including where the 137 MB weights file is absent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from retrieval_core.models import NOMIC_V15_INT8, ModelSpec, resolve

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from tokenizers import Tokenizer

__all__ = [
    "DOCUMENT_PREFIX",
    "QUERY_PREFIX",
    "Embedder",
    "HashingEmbedder",
    "NomicEmbedder",
    "TokenizerCounter",
    "pool_and_normalize",
]

# Exported so tests and documentation can name them. Note what is *not*
# exported: any function that takes one of these as an argument.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_DTYPE = np.float32

# PyTorch's `F.layer_norm` default, which is what nomic's reference
# post-processing uses. Deliberately not the model config's `layer_norm_epsilon`
# of 1e-12 — that governs the normalisation layers *inside* the transformer,
# which the ONNX graph already applied before it handed back hidden states.
_LAYER_NORM_EPS = 1e-5


@runtime_checkable
class Embedder(Protocol):
    """What storage and retrieval depend on, so a backend can be swapped.

    ``model_id`` and ``dimension`` are written into collection metadata and
    checked on every open (§6, frozen per collection). Two backends that produce
    incomparable vectors must never share a ``model_id``.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_context(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def pool_and_normalize(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    dimension: int | None = None,
) -> np.ndarray:
    """Mean-pool, layer-norm, truncate, L2-normalise — in that order.

    ``hidden_states`` is ``(batch, sequence, model_dim)`` and ``attention_mask``
    is ``(batch, sequence)`` with zeros on padding. Padding is excluded from the
    mean rather than averaged in as zeros: a short text in a batch of long ones
    would otherwise have its vector pulled toward the origin in proportion to
    how much padding it happened to receive, making a vector's meaning depend on
    its neighbours in the batch.

    ``dimension`` truncates (Matryoshka, §6); ``None`` keeps the full width.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be (batch, sequence, dim), got {hidden_states.shape}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {attention_mask.shape} does not match "
            f"hidden_states {hidden_states.shape[:2]}"
        )

    hidden = hidden_states.astype(np.float32, copy=False)
    mask = attention_mask.astype(np.float32, copy=False)[:, :, None]

    counted = mask.sum(axis=1)
    # An all-padding row has nothing to average. Clamping keeps it finite; the
    # result is the zero vector, which is the honest answer for empty input.
    pooled = (hidden * mask).sum(axis=1) / np.clip(counted, 1e-9, None)

    mean = pooled.mean(axis=1, keepdims=True)
    variance = pooled.var(axis=1, keepdims=True)
    normed = (pooled - mean) / np.sqrt(variance + _LAYER_NORM_EPS)

    if dimension is not None:
        if not 1 <= dimension <= normed.shape[1]:
            raise ValueError(f"dimension must be between 1 and {normed.shape[1]}, got {dimension}")
        normed = normed[:, :dimension]

    lengths = np.linalg.norm(normed, axis=1, keepdims=True)
    return (normed / np.clip(lengths, 1e-12, None)).astype(_DTYPE, copy=False)


class TokenizerCounter:
    """The model's own tokenizer, counting what the model will actually see.

    Replaces :class:`~retrieval_core.tokens.EstimatingCounter` once weights are
    available, and differs from it in one way worth stating: ``max_context``
    subtracts the task prefix and the special tokens. The caller's text is never
    what reaches the model — a prefix goes in front of it and ``[CLS]``/``[SEP]``
    wrap the result — so a chunk sized against the raw ceiling would be several
    tokens too long, and the tokenizer would silently drop its tail. Sizing
    against a ceiling the text can actually occupy is the difference between a
    chunk that is indexed and a chunk that is indexed minus its last sentence.
    """

    def __init__(self, tokenizer: Tokenizer, model_max_context: int) -> None:
        self._tokenizer = tokenizer
        prefix_cost = max(
            len(tokenizer.encode(prefix, add_special_tokens=True).ids)
            for prefix in (DOCUMENT_PREFIX, QUERY_PREFIX)
        )
        self._max_context = max(1, model_max_context - prefix_cost)

    @property
    def max_context(self) -> int:
        return self._max_context

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class NomicEmbedder:
    """nomic-embed-text-v1.5, INT8 ONNX, on the CPU.

    §7 makes embedding always local, so there is no remote path here and no key
    to expire. Construction is the expensive part — an ONNX session over 137 MB
    of weights — so build one and keep it.
    """

    def __init__(
        self,
        weights_path: Path,
        tokenizer_path: Path,
        *,
        spec: ModelSpec = NOMIC_V15_INT8,
        dimension: int | None = None,
        batch_size: int = 1,
    ) -> None:
        onnxruntime, tokenizers = _require_backend()

        if dimension is not None and not 1 <= dimension <= spec.dimension:
            raise ValueError(f"dimension must be between 1 and {spec.dimension}, got {dimension}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Above 1, a text's vector depends on what else was in its batch. See
        # the module docstring: this is a measured property of the INT8 export,
        # and raising it trades reproducible vectors for throughput.

        self._spec = spec
        self._dimension = dimension or spec.dimension
        self._batch_size = batch_size

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = onnxruntime.InferenceSession(
            str(weights_path), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {value.name for value in self._session.get_inputs()}

        self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=spec.max_context)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._counter = TokenizerCounter(self._tokenizer, spec.max_context)

    @classmethod
    def load(
        cls,
        *,
        spec: ModelSpec = NOMIC_V15_INT8,
        bundled_dir: Path | None = None,
        cache_dir: Path | None = None,
        allow_download: bool = False,
        dimension: int | None = None,
        batch_size: int = 1,
    ) -> NomicEmbedder:
        """Resolve weights (see :mod:`retrieval_core.models`) and open a session."""
        paths = resolve(
            spec,
            bundled_dir=bundled_dir,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        return cls(
            paths[spec.weights_file],
            paths[spec.tokenizer_file],
            spec=spec,
            dimension=dimension,
            batch_size=batch_size,
        )

    @property
    def model_id(self) -> str:
        # Truncated vectors are still this model's vectors; storage records the
        # dimension separately and checks it separately, so the id does not
        # need to carry it.
        return self._spec.model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_context(self) -> int:
        return self._counter.max_context

    @property
    def token_counter(self) -> TokenizerCounter:
        """Hand this to ``ChunkingConfig`` so chunks are sized by the real thing."""
        return self._counter

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text for indexing. Applies ``search_document:`` internally."""
        if not texts:
            return np.empty((0, self._dimension), dtype=_DTYPE)
        prefixed = [DOCUMENT_PREFIX + text for text in texts]
        return np.vstack([self._forward(batch) for batch in _batched(prefixed, self._batch_size)])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Applies ``search_query:`` internally.

        Returns a 1-D vector, not a batch of one: a query is a single thing, and
        a caller who has to remember to index ``[0]`` eventually forgets.
        """
        return self._forward([QUERY_PREFIX + text])[0]

    def _forward(self, prefixed: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(prefixed, add_special_tokens=True)
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )

        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # This export takes token_type_ids; other exports of the same model do
        # not. Feeding an input the graph does not declare is a hard error, so
        # supply it only when asked for.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        hidden_states = self._session.run(None, feeds)[0]
        return pool_and_normalize(hidden_states, attention_mask, self._dimension)


class HashingEmbedder:
    """A deterministic stand-in with no model behind it. **Tests only.**

    Increment 4 has to prove that hybrid retrieval blends dense and lexical
    scores correctly, that deletion removes vectors, that dimensions are
    enforced. None of that is about embedding quality, and all of it would
    otherwise need 137 MB of weights and several seconds per test.

    This is a bag-of-words hash, so similarity here tracks *word overlap* and
    nothing else — it has no idea that "invoice" and "receipt" are related. Never
    use it for anything a person will read the results of.

    The safeguard against that is ``model_id``: a collection built with this
    records ``hashing-test-v1``, and §6's frozen-model check then refuses to open
    it with the real embedder rather than serving nonsense from mixed vectors.
    Task prefixes are deliberately *not* applied — a constant token in every
    vector would put a similarity floor under every pair and flatter any
    retrieval test run against it.
    """

    def __init__(self, dimension: int = 64, max_context: int = 8192) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._max_context = max_context

    @property
    def model_id(self) -> str:
        return "hashing-test-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_context(self) -> int:
        return self._max_context

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=_DTYPE)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)[0]

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros((1, self._dimension), dtype=np.float64)
        for word in text.lower().split():
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            # The sign keeps unrelated words from stacking into the same
            # positive bucket, which would make everything look similar.
            vector[0, value % self._dimension] += 1.0 if value & 1 else -1.0
        length = np.linalg.norm(vector)
        if length == 0.0:
            return vector.astype(_DTYPE)
        return (vector / length).astype(_DTYPE)


def _batched(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _require_backend():  # noqa: ANN202 - returns two modules
    try:
        import onnxruntime
        import tokenizers
    except ImportError as error:  # pragma: no cover - exercised by absence, not by CI
        raise ImportError(
            "NomicEmbedder needs the 'embedding' extra: pip install retrieval-core[embedding]. "
            "Chunking and storage work without it."
        ) from error
    return onnxruntime, tokenizers
