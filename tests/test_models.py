"""Weight resolution, which exists to fail rather than to succeed.

The happy path here is one `shutil.copyfileobj` and a rename. Everything else in
this file is about the unhappy paths, because the failure mode that matters is
not "the download broke" — that one announces itself — it is "the file is
present, the wrong bytes are in it, and every embedding from here is subtly
wrong for as long as the index lives".

The download tests use `file://` URLs against a fake spec. That exercises the
real code — stream, hash, verify, atomically rename — without a 137 MB transfer
or a network CI depends on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from retrieval_core import models
from retrieval_core.models import (
    ModelFile,
    ModelSpec,
    WeightsCorruptError,
    WeightsMissingError,
    default_cache_dir,
    resolve,
)

PAYLOAD = b"not really a neural network, but the bytes are checked the same way"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def fake_spec(*files: ModelFile) -> ModelSpec:
    entries = files or (ModelFile(path="weights.bin", sha256=DIGEST, size=len(PAYLOAD)),)
    return ModelSpec(
        model_id="fake-model-v0",
        repo="nobody/fake-model",
        revision="0" * 40,
        files=entries,
        weights_file=entries[0].path,
        tokenizer_file=entries[0].path,
        dimension=8,
        max_context=128,
    )


def write(path: Path, payload: bytes = PAYLOAD) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def serve_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the downloader at a local directory instead of Hugging Face."""
    origin = tmp_path / "origin"
    origin.mkdir()
    template = origin.resolve().as_uri() + "/{path}"
    monkeypatch.setattr(models, "_HF_URL", template)
    return origin


class TestPinning:
    def test_the_real_spec_pins_a_commit_not_a_branch(self) -> None:
        # A revision of "main" would mean the weights can change under a
        # released product without the model_id changing with them.
        revision = models.NOMIC_V15_INT8.revision
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)

    def test_the_real_spec_pins_every_file_by_hash(self) -> None:
        assert models.NOMIC_V15_INT8.files
        for entry in models.NOMIC_V15_INT8.files:
            assert len(entry.sha256) == 64
            assert entry.size > 0

    def test_model_id_names_the_quantisation(self) -> None:
        # int8 and fp32 vectors are close but not identical; a collection that
        # mixed them would be quietly degraded, so the id has to separate them.
        assert "int8" in models.NOMIC_V15_INT8.model_id

    def test_file_lookup_by_path(self) -> None:
        spec = models.NOMIC_V15_INT8
        assert spec.file(spec.weights_file).size > 100_000_000
        with pytest.raises(KeyError):
            spec.file("onnx/model_fp32.onnx")


class TestCacheDir:
    def test_is_outside_any_collection(self, tmp_path: Path) -> None:
        # §5 requires a collection to be zippable and movable. A model cached
        # inside one would travel with every copy.
        assert tmp_path not in default_cache_dir().parents

    def test_is_absolute(self) -> None:
        assert default_cache_dir().is_absolute()


class TestResolveWithoutDownloading:
    def test_prefers_the_bundled_copy(self, tmp_path: Path) -> None:
        spec = fake_spec()
        bundled = tmp_path / "bundled"
        cache = tmp_path / "cache"
        write(bundled / "weights.bin")
        write(cache / spec.model_id / "weights.bin")

        resolved = resolve(spec, bundled_dir=bundled, cache_dir=cache)

        # A shipped product must use what it shipped, not whatever a developer
        # left in the cache on the same machine.
        assert resolved["weights.bin"] == bundled / "weights.bin"

    def test_falls_back_to_the_cache(self, tmp_path: Path) -> None:
        spec = fake_spec()
        cache = tmp_path / "cache"
        write(cache / spec.model_id / "weights.bin")

        resolved = resolve(spec, bundled_dir=tmp_path / "empty", cache_dir=cache)

        assert resolved["weights.bin"] == cache / spec.model_id / "weights.bin"

    def test_refuses_to_reach_the_network_by_default(self, tmp_path: Path) -> None:
        with pytest.raises(WeightsMissingError) as raised:
            resolve(fake_spec(), cache_dir=tmp_path / "cache")

        # The message has to say what to do, because the person reading it is a
        # developer on a fresh checkout who has no idea a download is optional.
        assert "allow_download=True" in str(raised.value)

    def test_names_every_place_it_looked(self, tmp_path: Path) -> None:
        with pytest.raises(WeightsMissingError) as raised:
            resolve(fake_spec(), bundled_dir=tmp_path / "bundled", cache_dir=tmp_path / "cache")

        message = str(raised.value)
        assert "bundled" in message
        assert "cache" in message

    def test_rejects_a_bundled_file_with_the_wrong_bytes(self, tmp_path: Path) -> None:
        spec = fake_spec()
        bundled = tmp_path / "bundled"
        write(bundled / "weights.bin", b"n" * len(PAYLOAD))

        with pytest.raises(WeightsCorruptError) as raised:
            resolve(spec, bundled_dir=bundled, cache_dir=tmp_path / "cache")

        # Same length, different content: the size check cannot catch this, and
        # this is the case that produces working-but-wrong embeddings.
        assert "hashes to" in str(raised.value)

    def test_rejects_a_truncated_file(self, tmp_path: Path) -> None:
        spec = fake_spec()
        cache = tmp_path / "cache"
        write(cache / spec.model_id / "weights.bin", PAYLOAD[:10])

        with pytest.raises(WeightsCorruptError) as raised:
            resolve(spec, cache_dir=cache)

        assert "bytes, expected" in str(raised.value)

    def test_verifies_on_every_resolve_not_just_the_first(self, tmp_path: Path) -> None:
        spec = fake_spec()
        cache = tmp_path / "cache"
        cached = write(cache / spec.model_id / "weights.bin")

        assert resolve(spec, cache_dir=cache)["weights.bin"] == cached

        # Disk corruption, a half-finished copy, a colleague's rsync. A
        # resolver that trusted its first look would sail past all of them.
        cached.write_bytes(b"x" * len(PAYLOAD))
        with pytest.raises(WeightsCorruptError):
            resolve(spec, cache_dir=cache)


class TestDownloading:
    def test_downloads_verifies_and_caches(self, tmp_path: Path, serve_locally: Path) -> None:
        spec = fake_spec()
        write(serve_locally / "weights.bin")
        cache = tmp_path / "cache"

        resolved = resolve(spec, cache_dir=cache, allow_download=True)

        assert resolved["weights.bin"].read_bytes() == PAYLOAD
        assert resolved["weights.bin"] == cache / spec.model_id / "weights.bin"

    def test_downloads_nested_paths(self, tmp_path: Path, serve_locally: Path) -> None:
        spec = fake_spec(ModelFile(path="onnx/model.onnx", sha256=DIGEST, size=len(PAYLOAD)))
        write(serve_locally / "onnx" / "model.onnx")

        resolved = resolve(spec, cache_dir=tmp_path / "cache", allow_download=True)

        assert resolved["onnx/model.onnx"].read_bytes() == PAYLOAD

    def test_downloads_only_what_is_missing(self, tmp_path: Path, serve_locally: Path) -> None:
        spec = fake_spec(
            ModelFile(path="weights.bin", sha256=DIGEST, size=len(PAYLOAD)),
            ModelFile(path="tokenizer.json", sha256=DIGEST, size=len(PAYLOAD)),
        )
        cache = tmp_path / "cache"
        write(cache / spec.model_id / "weights.bin")
        write(serve_locally / "tokenizer.json")
        # weights.bin is deliberately absent from the origin: if resolve tried
        # to fetch it despite the cache hit, this would fail to open.

        resolved = resolve(spec, cache_dir=cache, allow_download=True)

        assert len(resolved) == 2

    def test_a_corrupt_download_is_not_kept(self, tmp_path: Path, serve_locally: Path) -> None:
        spec = fake_spec()
        write(serve_locally / "weights.bin", b"served bytes that do not match the pin")
        cache = tmp_path / "cache"

        with pytest.raises(WeightsCorruptError):
            resolve(spec, cache_dir=cache, allow_download=True)

        # The whole point of downloading to a temporary name. A bad file left at
        # the cached name would be found — and, on the next run, trusted.
        assert not (cache / spec.model_id / "weights.bin").exists()

    def test_leaves_no_part_files_behind(self, tmp_path: Path, serve_locally: Path) -> None:
        spec = fake_spec()
        write(serve_locally / "weights.bin", b"wrong")
        cache = tmp_path / "cache"

        with pytest.raises(WeightsCorruptError):
            resolve(spec, cache_dir=cache, allow_download=True)

        assert list((cache / spec.model_id).glob("*.part")) == []

    def test_a_failed_download_leaves_nothing(self, tmp_path: Path, serve_locally: Path) -> None:
        spec = fake_spec()
        cache = tmp_path / "cache"

        with pytest.raises(OSError):
            resolve(spec, cache_dir=cache, allow_download=True)

        assert list((cache / spec.model_id).glob("*")) == []
