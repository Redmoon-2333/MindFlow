"""Tests for HMAC model-file signing/verification (F1 security hardening).

Covers:
  - sign_model_file / verify_model_file round-trip
  - tampered file fails verification
  - missing .hmac sibling fails verification
  - ModelManager.load_latest() refuses to load an unsigned/tampered model
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mindflow.train.models import ModelManager
from mindflow.train.models.manager import ModelSignatureError
from mindflow.train.serialization import (
    _load_or_create_signing_key,
    sign_model_file,
    verify_model_file,
)


@pytest.fixture
def tmp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestSignAndVerify:
    def test_round_trip_succeeds(self, tmp_dir: Path) -> None:
        """A freshly signed file should verify successfully."""
        model_path = tmp_dir / "model.pkl"
        model_path.write_bytes(b"fake pickle bytes")
        key = _load_or_create_signing_key(tmp_dir)

        sign_model_file(model_path, key)

        assert verify_model_file(model_path, key)
        assert (tmp_dir / "model.pkl.hmac").exists()

    def test_tampered_file_fails_verification(self, tmp_dir: Path) -> None:
        """Modifying the file after signing should invalidate the signature."""
        model_path = tmp_dir / "model.pkl"
        model_path.write_bytes(b"original content")
        key = _load_or_create_signing_key(tmp_dir)
        sign_model_file(model_path, key)

        model_path.write_bytes(b"tampered content!!")

        assert not verify_model_file(model_path, key)

    def test_missing_hmac_sibling_fails_verification(self, tmp_dir: Path) -> None:
        """A file with no .hmac sibling (never signed) must not verify."""
        model_path = tmp_dir / "model.pkl"
        model_path.write_bytes(b"never signed")
        key = _load_or_create_signing_key(tmp_dir)

        assert not verify_model_file(model_path, key)

    def test_wrong_key_fails_verification(self, tmp_dir: Path) -> None:
        """Verifying with a different key than the one used to sign must fail."""
        model_path = tmp_dir / "model.pkl"
        model_path.write_bytes(b"some content")
        key = _load_or_create_signing_key(tmp_dir)
        sign_model_file(model_path, key)

        other_dir = tmp_dir / "other"
        other_dir.mkdir()
        wrong_key = _load_or_create_signing_key(other_dir)

        assert not verify_model_file(model_path, wrong_key)

    def test_signing_key_persists_across_calls(self, tmp_dir: Path) -> None:
        """Calling _load_or_create_signing_key twice should return the same key."""
        key1 = _load_or_create_signing_key(tmp_dir)
        key2 = _load_or_create_signing_key(tmp_dir)
        assert key1 == key2
        assert len(key1) == 32


class TestModelManagerSignatureEnforcement:
    def test_load_latest_refuses_tampered_model(self, tmp_dir: Path) -> None:
        """After save_all + tampering a .pkl, load_latest must raise, not silently fail."""
        manager = ModelManager(models_dir=tmp_dir)
        rng = np.random.default_rng(0)
        n = 50
        features = rng.random((n, 14))
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < n // 2 else 0 for i in range(n)], dtype=np.int32)

        manager.train_all(features, feature_names, y)
        saved = manager.save_all()

        # Tamper with one saved model file after it was signed.
        tampered_path = tmp_dir / saved["clustering"]
        tampered_path.write_bytes(b"malicious pickle payload")

        manager2 = ModelManager(models_dir=tmp_dir)
        with pytest.raises(ModelSignatureError):
            manager2.load_latest()

    def test_load_latest_refuses_unsigned_legacy_model(self, tmp_dir: Path) -> None:
        """Model files saved before signing existed (no .hmac sibling) must be refused."""
        # Simulate a pre-F1 "legacy" install: .pkl files exist, latest.json
        # points at them, but there are no .hmac siblings at all.
        for name in ("clustering", "classifier", "hmm"):
            (tmp_dir / f"{name}-20260101.pkl").write_bytes(b"legacy unsigned model")
        (tmp_dir / "latest.json").write_text(
            json.dumps(
                {
                    "clustering": "clustering-20260101.pkl",
                    "classifier": "classifier-20260101.pkl",
                    "hmm": "hmm-20260101.pkl",
                }
            ),
            encoding="utf-8",
        )

        manager = ModelManager(models_dir=tmp_dir)
        with pytest.raises(ModelSignatureError):
            manager.load_latest()
