# Literature RAG

MindFlow retrieves reviewed research locally before generating explanations.
Personal behavior evidence and literature have separate citation namespaces:
`[证据: ID]` and `[论文: chunk_id]`.

The 2026-11-13 corpus includes 21 source documents and 27 page-verified
paraphrase cards. A private local index can additionally retrieve 187 chunks
from reviewed physical PDF pages. The public package contains original
paraphrases and provenance, not full paper PDFs.

See [the detailed Chinese guide](literature-rag.md) for paper findings, prompt
responsibilities, important prompt passages, retrieval implementation, and
evaluation boundaries. [The continuous presentation manuscript](defense-speech.md)
keeps the four-minute presentation separate from technical backup material.

The [campus pilot and delivery plan](landing-plan.md) covers first users,
local deployment, installation acceptance, planned pilot tasks, evaluation,
operating costs, maintenance, and adoption. It distinguishes the running
prototype from future recruitment and user research.

The [acceptance record](reports/acceptance-20261113.md) separates the published
baseline's 3477 passing tests from the local workspace's 3500 passing tests.
Both have 4 skipped tests and 22 warnings. Retrieval diagnostics are
development checks, not a user-effectiveness study.

## Runtime

The default retriever uses SQLite FTS5/BM25 with a reviewed bilingual glossary,
topic and role filtering, one passage per paper, and bounded context.
`MINDFLOW_RESEARCH_INDEX_PATH` can select a private read-only index. Without
that index, the packaged cards are indexed in memory and remain usable offline.

Panel, chat, intervention, structured single-expert, and Ollama fallback
generation paths receive retrieved context. Generated literature IDs must
belong to the exact context supplied for that generation. A moderator or critic
also receives passages cited upstream in the same panel run.

## Rebuilding With Local PDFs

Download lawfully accessible PDFs from the recorded source URLs into a local
directory. Preserve the filenames and verify hashes against
`sources/download-manifest.json`. This manifest contains metadata only.
Place a copy named `download-manifest.json` alongside the PDFs, then run from
`backend-next`:

```powershell
uv run --with pymupdf python scripts/build_literature_index.py `
  --paper-dir <local-pdf-directory> `
  --annotations ../docs/research/annotations/psychology_annotations.json `
    ../docs/research/annotations/hci_annotations.json `
    ../docs/research/annotations/rag_annotations.json

uv run python scripts/evaluate_literature_retrieval.py
uv run python -m pytest tests/test_literature_retrieval.py -q
```

The build rejects source hash, page count, and anchor mismatches. It indexes
only reviewed physical pages, including the specifically reviewed manuscript
pages inside the Eckert thesis carrier. Source-version metadata identifies
that carrier; it is not the journal version of record.

The query report is a development diagnostic with declared acceptable sources,
not an independent benchmark. Citation-ID validity does not establish semantic
support, and retrieval results do not establish improved attention or task
outcomes. Those require separate blinded output review and user research.
