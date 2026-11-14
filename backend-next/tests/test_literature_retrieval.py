"""Behavioral checks for retrieval, provenance and the real prompt entry points."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage

from mindflow.agents.experts import MODERATOR, TMT
from mindflow.graph.chat_graph import _build_messages_from_state, answer_extraction_node
from mindflow.graph.panel_graph import _call_with_budget, _PanelRunContext
from mindflow.services.intervention_service import _build_message_user_content
from mindflow.services.literature_service import (
    LiteratureCitationError,
    create_index,
    invalid_literature_citations,
    retrieve_literature,
    reviewed_passages,
    search_terms,
)


def test_packaged_sources_have_traceable_pages() -> None:
    rows = reviewed_passages()
    assert len(rows) >= 15
    assert len({row["paper_id"] for row in rows}) >= 10
    assert all(len(row["pdf_sha256"]) == 64 and row["pages"] for row in rows)
    assert all(row["source_kind"] == "reviewed_paraphrase" for row in rows)
    assert all(row["application"] and row["limits"] for row in rows)


@pytest.mark.parametrize("query,expected", [
    ("动机期望价值截止", "steel"),
    ("情绪调节任务厌恶", "sirois"),
    ("AI编程探索切换", "barke"),
    ("提醒时机接受打扰", "nahum"),
    ("引用核验citation", "gao"),
])
def test_bilingual_queries_find_relevant_papers(query: str, expected: str, tmp_path: Path) -> None:
    result = retrieve_literature(query, role="chat", index_path=tmp_path / "absent.sqlite3")
    assert any(expected in hit.paper_id for hit in result.hits), result.hits
    assert result.backend == "reviewed_fts5"


@pytest.mark.parametrize("query", ["", "今天天气怎样", "quantum chromodynamics", '" OR 1=1 --'])
def test_unrelated_and_fts_syntax_queries_do_not_invent_context(query: str) -> None:
    assert not retrieve_literature(query, index_path=Path("absent.sqlite3")).hits


def test_query_has_no_raw_sql_fts_syntax() -> None:
    assert all(token.replace("_", "").isalnum() for token in search_terms('rag" OR NEAR(foo)'))


def test_missing_or_corrupt_pdf_index_uses_reviewed_cards(tmp_path: Path) -> None:
    index = tmp_path / "corrupt.sqlite3"
    index.write_text("not a SQLite file")
    result = retrieve_literature("emotion regulation", index_path=index)
    assert result.hits
    assert result.backend == "reviewed_fts5"


def test_pdf_index_is_used_and_preserves_physical_page(tmp_path: Path) -> None:
    row = dict(reviewed_passages()[0])
    row.update(source_kind="pdf_excerpt", content="temporal motivation expectancy value",
               terms="temporal motivation", pages=[9], chunk_id="fixture.p009.c01",
               roles=["chat"])
    path = tmp_path / "index.sqlite3"
    with sqlite3.connect(path) as connection:
        create_index(connection, [row])
    result = retrieve_literature("temporal motivation", index_path=path)
    assert result.backend == "pdf_fts5"
    assert result.hits[0].pages == (9,)
    assert result.hits[0].source_kind == "pdf_excerpt"


def test_role_filter_and_context_budget(tmp_path: Path) -> None:
    result = retrieve_literature(
        "motivation", role="TMT归因专家", index_path=tmp_path / "missing", max_chars=3000,
    )
    assert result.hits
    assert len(result.as_json()) < 3300
    assert len({hit.paper_id for hit in result.hits}) == len(result.hits)
    assert all("gao" not in hit.paper_id and "lewis" not in hit.paper_id for hit in result.hits)


def test_generic_task_word_does_not_fill_programming_query_with_ir_papers() -> None:
    result = retrieve_literature("AI编程探索切换")
    assert any(hit.paper_id == "barke_2023" for hit in result.hits)
    assert all(hit.paper_id != "robertson_2009" for hit in result.hits)


def test_literature_ids_are_not_behavior_metric_ids() -> None:
    assert invalid_literature_citations("[论文: fake.p001.c01]", {"focus.switch_rate"})
    assert not invalid_literature_citations("[证据: focus.switch_rate]", set())
    assert invalid_literature_citations("[论文: ]", set()) == ("malformed_citation",)


@pytest.mark.asyncio
async def test_panel_actual_gateway_receives_retrieval_and_rejects_fabrication() -> None:
    class Gateway:
        async def complete(self, system: str, user: str, model: str = "chat") -> str:
            assert "文献与个人证据" in system
            context = json.loads(user.split("## research_context\n")[1])
            assert context["passages"]
            assert any("steel" in row["paper_id"] for row in context["passages"])
            return '{"argument":"[论文: nonexistent.p001.c01]"}'

    with pytest.raises(LiteratureCitationError):
        await _call_with_budget(_PanelRunContext(), Gateway(), TMT, '{"evidence_catalog":[]}')


@pytest.mark.asyncio
async def test_moderator_receives_upstream_cited_passage() -> None:
    class Gateway:
        citation = ""

        async def complete(self, system: str, user: str, model: str = "chat") -> str:
            context = json.loads(user.split("## research_context\n")[1])
            assert context["passages"]
            if not self.citation:
                self.citation = context["passages"][0]["chunk_id"]
            assert self.citation in {row["chunk_id"] for row in context["passages"]}
            return f"已核对 [论文: {self.citation}]"

    gateway = Gateway()
    runtime = _PanelRunContext()
    opinion = await _call_with_budget(runtime, gateway, TMT, "动机期望价值截止")
    await _call_with_budget(runtime, gateway, MODERATOR, opinion)


@pytest.mark.asyncio
async def test_chat_does_not_publish_fabricated_citation() -> None:
    state = {
        "user_id": 1, "session_id": "research-test",
        "user_message": "如何理解情绪调节和拖延？", "messages": [],
        "model_messages_raw": [AIMessage(content="结论 [论文: fake.p001.c01]")],
    }
    messages = _build_messages_from_state(state)
    assert "research_context" in messages[0].content
    outcome = await answer_extraction_node(state)
    assert outcome["degraded"] is True
    assert "fake" not in outcome["answer"]


def test_intervention_actual_builder_includes_receptivity_research() -> None:
    prompt = _build_message_user_content("{}", "nudge", "gentle")
    context = json.loads(prompt.split("## research_context\n")[1])
    assert context["passages"]
    assert any("nahum" in row["paper_id"] for row in context["passages"])


@pytest.mark.asyncio
async def test_chat_keeps_the_context_used_for_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from mindflow.graph.chat_graph import ChatRunContext, model_call_node
    from mindflow.services.literature_service import LiteratureContext

    class Model:
        async def ainvoke(self, messages: list[AIMessage]) -> AIMessage:
            content = str(messages[0].content)
            context = json.loads(content.split("## research_context\n")[1])
            citation = context["passages"][0]["chunk_id"]
            return AIMessage(content=f"可供核对的解释 [论文: {citation}]")

    state = {
        "user_id": 1, "session_id": "context-stability",
        "user_message": "情绪调节和拖延", "runtime": ChatRunContext(model=Model()),
    }
    state.update(await model_call_node(state))
    monkeypatch.setattr(
        "mindflow.graph.chat_graph.retrieve_literature", lambda *a, **kw: LiteratureContext(),
    )
    result = await answer_extraction_node(state)
    assert "[论文:" in result["answer"]
    assert not result.get("degraded")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["single", "ollama", "intervention"])
@pytest.mark.parametrize("fabricate", [False, True])
async def test_http_entry_points_use_the_same_citation_allowlist(
    path: str, fabricate: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mindflow.config import LLMSettings
    from mindflow.graph.fallback_nodes import _ollama_api_call
    from mindflow.infrastructure.llm.client import DeepSeekClient
    from mindflow.services.intervention_service import _generate_llm_message

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "文献与个人证据" in body["messages"][0]["content"]
        context = json.loads(body["messages"][1]["content"].split("## research_context\n")[1])
        assert context["passages"]
        citation = "fabricated.p001.c01" if fabricate else context["passages"][0]["chunk_id"]
        calls.append(citation)
        if path == "intervention":
            answer = {"title": "先写下一步", "message": f"可以先选一步。[论文: {citation}]"}
        else:
            answer = {
                "procrastination_types": ["task_aversion"],
                "type_confidence": {"task_aversion": 0.2},
                "cognitive_distortions": [],
                "cbt_technique": "goal_setting",
                "response_text": f"这个解释需要核对。[论文: {citation}]",
                "next_action": "说说任务目标",
            }
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(answer, ensure_ascii=False)}}],
        })

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://test.example",
    ) as transport:
        if path == "intervention":
            outcome = await _generate_llm_message(transport, "test", "{}", "nudge", "gentle")
            assert (outcome is None) is fabricate
        else:
            single = None
            if path == "single":
                single = DeepSeekClient(LLMSettings(
                    api_key="fixture-key", base_url="https://test.example",
                    model="fixture", ecnu_compat_enabled=True,
                ))
                await single._client.aclose()
                single._client = transport
            if fabricate:
                with pytest.raises(LiteratureCitationError):
                    if single is not None:
                        await single.analyze("{}")
                    else:
                        await _ollama_api_call(
                            "https://test.example", "fixture", "{}", client=transport,
                        )
            elif single is not None:
                assert calls == []
                await single.analyze("{}")
            else:
                await _ollama_api_call("https://test.example", "fixture", "{}", client=transport)
    assert len(calls) == 1
