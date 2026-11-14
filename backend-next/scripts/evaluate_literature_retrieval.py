"""Small declared-query diagnostic; no generation, human trial or paid calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from mindflow.services.literature_service import retrieve_literature

BACKEND = Path(__file__).resolve().parents[1]
CASES = (
    ("动机期望价值截止", "chat", ("steel_konig_2006",)),
    ("情绪调节任务厌恶", "chat", ("sirois_pychyl_2013", "tong_2024", "eckert_2016")),
    ("AI编程探索切换", "chat", ("barke_2023",)),
    ("提醒时机接受打扰", "chat", ("nahum_shani_2018", "mark_2008")),
    ("引用核验citation", "chat", ("gao_2023_alce",)),
    ("temporal motivation expectancy", "TMT归因专家", ("steel_konig_2006",)),
    ("task aversiveness", "情绪调节归因专家", ("sirois_pychyl_2013", "tong_2024")),
    ("receptivity fatigue", "intervention", ("nahum_shani_2018",)),
    ("多角色强单角色对照", "批评家", ("wang_2024",)),
    ("今天天气怎样", "chat", ()),
    ("quantum chromodynamics", "chat", ()),
    ('" OR 1=1 --', "chat", ()),
)


def evaluate(index_path: Path) -> dict[str, object]:
    rows = []
    for query, role, expected in CASES:
        started = time.perf_counter()
        context = retrieve_literature(query, role=role, index_path=index_path)
        ranked = [hit.paper_id for hit in context.hits]
        rank = next((i + 1 for i, value in enumerate(ranked) if value in expected), None)
        rows.append({
            "query": query, "role": role, "acceptable_paper_ids": expected,
            "paper_ids": ranked, "chunk_ids": [hit.chunk_id for hit in context.hits],
            "pages": [hit.pages for hit in context.hits], "backend": context.backend,
            "hit_at_3": rank is not None if expected else None,
            "reciprocal_rank": 1 / rank if rank else 0,
            "negative_pass": not ranked if not expected else None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        })
    positives = [row for row in rows if row["hit_at_3"] is not None]
    negatives = [row for row in rows if row["negative_pass"] is not None]
    return {
        "positive_cases": len(positives), "negative_cases": len(negatives),
        "hit_at_3": sum(bool(row["hit_at_3"]) for row in positives) / len(positives),
        "mrr_at_3": sum(float(row["reciprocal_rank"]) for row in positives) / len(positives),
        "negative_passed": sum(bool(row["negative_pass"]) for row in negatives),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=BACKEND / "data/experiments/20261113_literature_rag",
    )
    args = parser.parse_args()
    corpus = BACKEND / "src/mindflow/resources/literature/corpus.json"
    result = {
        "date": "2026-11-13",
        "scope": "Declared development queries; not a held-out benchmark or semantic evaluation.",
        "llm_calls": 0, "human_participants": 0,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "packaged": evaluate(args.output_dir / "absent-index.sqlite3"),
        "local_pdf": evaluate(BACKEND / "data/literature/index.sqlite3"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        name: {key: value for key, value in result[name].items() if key != "cases"}
        for name in ("packaged", "local_pdf")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
