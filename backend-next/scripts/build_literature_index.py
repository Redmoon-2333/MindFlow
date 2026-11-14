"""Verify source PDFs and build a private page-chunk index plus reviewed cards.

Run with ``uv run --with pymupdf python scripts/build_literature_index.py``.
Full PDF text stays under data/; only original paraphrases enter package data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pymupdf

from mindflow.services.literature_service import create_index

BACKEND = Path(__file__).resolve().parents[1]


def normalized(text: str) -> str:
    return re.sub(r"\W+", "", text.casefold())


def page_chunks(text: str, size: int = 1200, overlap: int = 160) -> list[str]:
    """Keep each chunk inside one physical PDF page for unambiguous citations."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 80:
        return []
    return [
        text[start:start + size]
        for start in range(0, len(text), size - overlap)
        if len(text[start:start + size].strip()) >= 80
    ]


def build(paper_dir: Path, annotation_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    entries = json.loads((paper_dir / "download-manifest.json").read_text("utf-8"))
    manifest = {entry["id"]: entry for entry in entries if entry.get("status") == "verified"}
    annotations: list[dict[str, Any]] = []
    for path in annotation_paths:
        annotations.extend(json.loads(path.read_text("utf-8-sig")))
    if not annotations:
        raise ValueError("No reviewed annotations supplied")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in annotations:
        grouped.setdefault(card["paper_id"], []).append(card)

    reviewed: list[dict[str, Any]] = []
    fulltext: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for paper_id, cards in grouped.items():
        paper = manifest[paper_id]
        path = paper_dir / paper["filename"]
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if not blob.startswith(b"%PDF-") or digest != paper["pdf_sha256"]:
            raise ValueError(f"Source header/hash mismatch: {paper_id}")
        with pymupdf.open(path) as pdf:
            if not pdf.is_pdf or len(pdf) != paper["pages"]:
                raise ValueError(f"Source page count mismatch: {paper_id}")
            page_texts = [page.get_text(sort=True) for page in pdf]
        if sum(map(len, page_texts)) < 500:
            raise ValueError(f"Source has no usable text: {paper_id}")
        # Metadata and our paraphrases are public; the PDF and full text are private.
        identifier = paper.get("doi") or (
            f"arXiv:{paper['arxiv']}" if paper.get("arxiv") else paper["url"]
        )
        base = {
            "paper_id": paper_id,
            "title": paper["title"],
            "doi": identifier,
            "pdf_sha256": digest,
        }
        for card in cards:
            pages = card["pages"]
            if not pages or any(type(page) is not int or not 1 <= page <= len(page_texts)
                                for page in pages):
                raise ValueError(f"Invalid physical PDF pages: {paper_id}/{card['key']}")
            original = "\n".join(page_texts[page - 1] for page in pages)
            if normalized(card["anchor"]) not in normalized(original):
                raise ValueError(f"Anchor missing on declared pages: {paper_id}/{card['key']}")
            reviewed.append({
                **base,
                "chunk_id": f"{paper_id}.p{pages[0]:03d}.{card['key']}",
                "pages": pages,
                "source_kind": "reviewed_paraphrase",
                "content": card["summary"],
                "application": card["application"],
                "limits": card["limits"],
                "terms": card["terms"],
                "roles": card["roles"],
                "anchor": card["anchor"],
            })
        # Index reviewed pages only; thesis carriers and reference lists are not
        # silently treated as the cited article or as reviewed research.
        selected_pages = sorted({page for card in cards for page in card["pages"]})
        for page_number in selected_pages:
            text = page_texts[page_number - 1]
            page_cards = [card for card in cards if page_number in card["pages"]]
            roles = sorted({role for card in page_cards for role in card["roles"]})
            terms = " ".join(card["terms"] for card in page_cards)
            for index, chunk in enumerate(page_chunks(text), 1):
                fulltext.append({
                    **base,
                    "chunk_id": f"{paper_id}.p{page_number:03d}.c{index:02d}",
                    "pages": [page_number],
                    "source_kind": "pdf_excerpt",
                    "content": chunk,
                    "application": page_cards[0]["application"],
                    "limits": page_cards[0]["limits"],
                    "terms": terms,
                    "roles": roles,
                    "page_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                })
        sources.append({
            **base,
            "authors": paper["authors"], "year": paper["year"],
            "filename": paper["filename"], "url": paper["url"],
            "download_url": paper.get("final_download_url"),
            "pages": paper["pages"], "bytes": paper["bytes"],
            "download_status": paper["download_status"],
            "license": paper["license"],
            "source_version": paper.get("source_version", "downloaded_research_copy"),
            "indexed_pdf_pages": selected_pages,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    resource_dir = BACKEND / "src/mindflow/resources/literature"
    resource_dir.mkdir(parents=True, exist_ok=True)
    corpus = {
        "schema_version": 1, "reviewed_on": "2026-11-13",
        "scope": "Original Chinese paraphrases; page-verified; not original PDF quotations.",
        "papers": sources, "passages": reviewed,
    }
    (resource_dir / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    public_manifest = [{**source, "status": "verified"} for source in sources]
    (output_dir / "download-manifest.public.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    # This build owns only its data-directory index. Rebuilding never touches user DBs.
    index_path = output_dir / "index.sqlite3"
    with sqlite3.connect(index_path) as connection:
        connection.execute("DROP TABLE IF EXISTS literature")
        create_index(connection, [*reviewed, *fulltext])
    report = {
        "date": "2026-11-13", "papers": len(sources),
        "reviewed_cards": len(reviewed), "pdf_chunks": len(fulltext),
        "chunk_chars": 1200, "overlap_chars": 160, "cross_page_chunks": 0,
        "index_scope": "reviewed_physical_pdf_pages_only",
        "index_bytes": index_path.stat().st_size,
        "corpus_sha256": hashlib.sha256((resource_dir / "corpus.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "verified_source_ids": sorted(grouped),
        "source_hash_and_page_anchor_checks": "PASS",
    }
    (output_dir / "build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=BACKEND / "data/literature")
    args = parser.parse_args()
    print(json.dumps(build(args.paper_dir, args.annotations, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
