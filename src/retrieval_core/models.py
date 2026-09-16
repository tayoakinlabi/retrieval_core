"""Finding the model weights, and refusing to use the wrong ones.

§6 resolved L4 as *bundle the weights in the installer*, which settles it for the
three products but not for the library: 137 MB cannot live in git, and a
developer checking out this repo has to get the file from somewhere. So there
are two paths and they are deliberately asymmetric.

**Bundled first, downloaded only when asked.** ``resolve`` looks in the
directory a product hands it — the installer's payload — before it looks
anywhere else, and it will not reach the network unless the caller passes
``allow_download=True``. A shipped product never passes it. An offline install
that silently started downloading a model would break the privacy claim the
portfolio is built on (§7: embedding is always local), and it would do so on the
one machine least able to notice.

**Every file is hashed, every time.** Not the size, not a stamp file written
beside it — the SHA-256 of the bytes, on the bundled path as well as the cached
one. Hashing 137 MB costs a fraction of a second against ONNX session creation,
and the failure it prevents is the quiet kind: a truncated or swapped weights
file does not crash, it produces embeddings that are merely *wrong*, and an
index built from them looks completely normal until someone notices retrieval
has been useless for a month.

The revision is pinned to a commit, not to ``main``. Hashes below were read from
the Hugging Face API at that commit; a repo that moves under us fails the hash
check rather than quietly changing the model.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "NOMIC_V15_INT8",
    "ModelFile",
    "ModelSpec",
    "WeightsCorruptError",
    "WeightsMissingError",
    "default_cache_dir",
    "resolve",
]

_HF_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
_CHUNK = 1 << 20


class WeightsMissingError(RuntimeError):
    """Weights are not bundled, not cached, and downloading was not permitted."""


class WeightsCorruptError(RuntimeError):
    """A weights file exists but its contents are not what was pinned."""


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One file of a model, pinned by content rather than by name."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to obtain and identify one embedding model.

    ``model_id`` is written into collection metadata and checked on every open
    (§6, "frozen per collection"), so it names the *weights* — quantisation
    included — not just the model family. Vectors from the int8 export and the
    fp32 export are close but not identical, and a collection that mixed them
    would have a silently degraded neighbourhood around every boundary.
    """

    model_id: str
    repo: str
    revision: str
    files: tuple[ModelFile, ...]
    weights_file: str
    tokenizer_file: str
    dimension: int
    max_context: int

    def file(self, path: str) -> ModelFile:
        for candidate in self.files:
            if candidate.path == path:
                return candidate
        raise KeyError(path)


# Hashes and sizes read from the Hugging Face API at the pinned revision.
# `onnx/model_int8.onnx` is 137,296,292 bytes — §6 estimated "~132 MB as INT8
# ONNX" from another project's measurement, which lands within a megabyte of
# 130.9 MiB. The identical-hash sibling `model_quantized.onnx` is the same bytes
# under an older name; either would do, and the explicit one is clearer.
NOMIC_V15_INT8 = ModelSpec(
    model_id="nomic-embed-text-v1.5-int8",
    repo="nomic-ai/nomic-embed-text-v1.5",
    revision="e9b6763023c676ca8431644204f50c2b100d9aab",
    files=(
        ModelFile(
            path="onnx/model_int8.onnx",
            sha256="b4342336debaea79de872370664b0aaeb67dea4605513d00ee236ea871a81f27",
            size=137296292,
        ),
        ModelFile(
            path="tokenizer.json",
            sha256="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            size=711396,
        ),
    ),
    weights_file="onnx/model_int8.onnx",
    tokenizer_file="tokenizer.json",
    dimension=768,
    max_context=2048,
)


def default_cache_dir() -> Path:
    """Where a developer checkout caches downloaded weights.

    Not inside the repo and not inside a collection: a collection is required to
    be zippable and movable (§5), and a 137 MB model that travelled with every
    copy would make that promise expensive. Products override this with their
    installed payload directory and never touch the cache at all.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "retrieval-core" / "models"
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "retrieval-core" / "models"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: ModelFile) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected.size:
        raise WeightsCorruptError(
            f"{path} is {actual_size} bytes, expected {expected.size}. "
            "A partial download, or a different build of the model."
        )
    actual = _digest(path)
    if actual != expected.sha256:
        raise WeightsCorruptError(
            f"{path} hashes to {actual}, expected {expected.sha256}. "
            "These are not the pinned weights; embeddings from them would be wrong."
        )


def _download(spec: ModelSpec, wanted: ModelFile, destination: Path) -> None:
    """Fetch one file, verify it, and only then put it where it will be found.

    The download lands in a temporary file in the destination directory and is
    renamed once the hash matches. An interrupted download must never be left at
    the cached name: the next run would find a file of the right name and, on a
    machine where the hash check was skipped for speed, trust it.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _HF_URL.format(repo=spec.repo, revision=spec.revision, path=wanted.path)
    handle, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, out, _CHUNK)
        _verify(temporary, wanted)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def resolve(
    spec: ModelSpec = NOMIC_V15_INT8,
    *,
    bundled_dir: Path | None = None,
    cache_dir: Path | None = None,
    allow_download: bool = False,
) -> dict[str, Path]:
    """Return a verified local path for every file in ``spec``.

    Searched in order: ``bundled_dir`` (what a product installs), then
    ``cache_dir`` (what a developer downloads). Raises rather than downloading
    unless ``allow_download`` is set, so a product cannot acquire a network
    dependency by accident.
    """
    cache_dir = cache_dir or default_cache_dir()
    resolved: dict[str, Path] = {}

    for wanted in spec.files:
        if bundled_dir is not None:
            candidate = bundled_dir / wanted.path
            if candidate.exists():
                _verify(candidate, wanted)
                resolved[wanted.path] = candidate
                continue

        cached = cache_dir / spec.model_id / wanted.path
        if cached.exists():
            _verify(cached, wanted)
            resolved[wanted.path] = cached
            continue

        if not allow_download:
            searched = [str(cached)]
            if bundled_dir is not None:
                searched.insert(0, str(bundled_dir / wanted.path))
            raise WeightsMissingError(
                f"{spec.model_id} file {wanted.path!r} not found in {', '.join(searched)}. "
                "Pass allow_download=True to fetch it, or point bundled_dir at the "
                "installed model directory."
            )

        _download(spec, wanted, cached)
        resolved[wanted.path] = cached

    return resolved
