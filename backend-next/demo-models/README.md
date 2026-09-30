# Synthetic ML Demo Bundle

The `v4/` directory contains a pretrained MindFlow feature-schema-v4 model.
It is a behavior classifier, not a language model.

## Provenance

- Source: only the project's synthetic student-archetype generator.
- Recipe: 6 archetypes, 14 days per archetype, seed 42.
- Features: schema v4, 28 ordered numeric features.
- Training pipeline: `run_training(source="synthetic_v2", allow_activation=False)`.
- No personal database, window titles, browsing records, raw training rows,
  API tokens, or local signing keys are distributed.
- The classifier, clustering model, and research-only HMM are included to
  preserve the existing `ModelManager` artifact layout.

`v4/bundle.json` records each artifact's SHA256, size, feature order, and exact
package versions. Its own digest is pinned in `scripts/demo.py`; editing an
artifact and merely replacing its checksum in the manifest is not sufficient.
The installer checks the release bytes before generating machine-local HMAC
signatures. It never disables the regular signed-model loading checks.
Git attributes preserve manifest/artifact bytes even when the target computer
uses different newline settings. The ML extra pins the manifest's serialization
dependencies; updating them requires rebuilding and revalidating the bundle.

## Use

Build `frontend/` with `npm ci` and `npm run build`. From `backend-next/`:

```powershell
uv sync --locked --extra dev --extra ml
uv run python scripts/demo.py
```

In another terminal, run `uv run python scripts/demo.py --login` and open the
one-time local URL. The combined frontend/backend listens on port 8870.
Use `--port` to choose a different unused port, including on the login command.

The separate runtime directory is `backend-next/data/demo/`. The launcher
refuses an existing directory without the demo marker, preserves existing
demo-page edits and model versions, and refreshes synthetic inference windows.
Collectors, scheduling, and online model credentials are off by default.

## Interpretation

The demo model is loadable and can return real inference probabilities.
It remains labelled `shadow` / `demo_only`: synthetic-data performance is not
evidence that a personal model passed its deployment quality gates.
The application can show predictions and populated reports without waiting
for collection or retraining. Cloud AI functions use the ordinary offline
fallback, not a bundled generative language model.
With collection disabled, the initial synthetic windows expire after 15 minutes
under the normal prediction freshness check. Restart the demo to refresh them;
the UI reports expired data instead of presenting a stale probability.

The original personal models remain private and are not replaced. Old
scikit-learn artifacts require their original environment or a separate
current-environment retraining; do not suppress version or signature checks.
