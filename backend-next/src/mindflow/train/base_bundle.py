"""Verify and install the application-owned, synthetic cold-start model.

The release manifest digest is pinned in code. We verify its exact artifact
bytes before local signing; ordinary personal models keep the existing HMAC
loader and active pointer. An install never writes into the personal v2 store.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.train.models.manager import ModelManager
from mindflow.train.serialization import _load_or_create_signing_key, sign_model_file

BASE_VERSION = "base_v4_20261111_r2"
MANIFEST_SHA256 = "73708e66f471bf3947bce32d13fc768c8168262d58f85596ff4c976cab8f05a8"
BUNDLE_DIR = Path(__file__).resolve().parents[1] / "resources" / "base-v4"
KINDS = ("clustering", "classifier", "hmm")


def verified_base_bundle(bundle: Path | None = None) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read once, verify before unpickling, and return the verified byte strings."""
    root = BUNDLE_DIR if bundle is None else bundle
    raw_manifest = (root / "bundle.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Base release manifest checksum differs from this application release")
    manifest: dict[str, Any] = json.loads(raw_manifest)
    if (
        manifest.get("version") != BASE_VERSION
        or manifest.get("source") != "synthetic_base"
        or manifest.get("synthetic_only") is not True
        or manifest.get("personal_quality_gate_passed") is not False
        or manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or manifest.get("feature_names") != list(V2_FEATURE_NAMES)
    ):
        raise ValueError("Base release provenance or feature schema is incompatible")
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(
                f"Base release requires {name}=={version}; run uv sync --locked --extra ml"
            )
    artifacts: dict[str, bytes] = {}
    for kind in KINDS:
        info = manifest["files"][kind]
        if info["name"] != f"{kind}-{BASE_VERSION}.pkl":
            raise ValueError("Unexpected base artifact name")
        raw = (root / info["name"]).read_bytes()
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError(f"Base release artifact checksum mismatch: {kind}")
        artifacts[kind] = raw
    return manifest, artifacts


def load_bundled_base_model(models_dir: Path, *, bundle: Path | None = None) -> ModelManager:
    """Install to a dedicated cache and load without touching personal pointers.

    A staging directory plus rename avoids partially installed releases. Each
    process verifies the packaged bytes and the cached bytes, even on a repeat
    launch. Copying only verified bytes prevents a verification/copy race.
    """
    manifest, artifacts = verified_base_bundle(bundle)
    cache_root = models_dir / "bundled"
    target = cache_root / BASE_VERSION
    names = {kind: manifest["files"][kind]["name"] for kind in KINDS}
    if not target.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".base-stage-", dir=cache_root))
        try:
            key = _load_or_create_signing_key(staging)
            for kind, raw in artifacts.items():
                artifact = staging / names[kind]
                artifact.write_bytes(raw)
                sign_model_file(artifact, key)
            (staging / "latest.json").write_text(json.dumps(names) + "\n", encoding="utf-8")
            (staging / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            probe = ModelManager(staging, use_ensemble=False)
            if not probe.load_latest() or not probe.readiness_status()["ready"]:
                raise ValueError("Base release failed its signed trial load")
            try:
                os.rename(staging, target)
            except OSError:
                if not target.is_dir():
                    raise
                # A concurrent startup may have installed the same release;
                # the target is checked byte-for-byte below before loading.
        finally:
            if staging.exists():
                import shutil

                shutil.rmtree(staging)
    for kind, expected in artifacts.items():
        if (target / names[kind]).read_bytes() != expected:
            raise ValueError(f"Cached base artifact checksum mismatch: {kind}")
    key_path = target / "model_signing.key"
    if not key_path.is_file() or key_path.stat().st_size != 32:
        raise ValueError("Cached base release has no valid local signing key")
    manager = ModelManager(target, use_ensemble=False)
    # Do not trust an altered cache pointer: load the manifest-pinned version.
    if not manager.load_version(BASE_VERSION) or not manager.readiness_status()["ready"]:
        raise ValueError("Base release could not be loaded with local signature verification")
    if manager.classifier.feature_names_ != list(V2_FEATURE_NAMES):
        raise ValueError("Loaded base classifier has incompatible feature order")
    manager.model_origin = "bundled_base"
    return manager
