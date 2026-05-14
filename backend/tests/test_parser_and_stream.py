from pathlib import Path
from types import SimpleNamespace

from app.agents.nodes.auditor import FaithfulnessScore
from app.agents.nodes import retriever_node as retriever_node_module
from app.api import routes
from app.config import Settings
from app.knowledge import parser as parser_module
from app.knowledge.indexer import ZhipuEmbeddingFunction
from app.knowledge.parser import DualTrackMedicalParser
from app.knowledge.retriever import MultiGranularityRetriever
from app.models.schemas import IntentType, TrustLevel
from rebuild_index import _build_index_status


def test_track_a_text_uses_page_number_metadata(monkeypatch):
    def fake_to_markdown(*args, **kwargs):
        return [
            {
                "metadata": {"page_number": 4},
                "text": "CAP：年龄3个月以上：10mg/kg 静脉滴注，qd，至少2天。",
            }
        ]

    monkeypatch.setattr(parser_module.pymupdf4llm, "to_markdown", fake_to_markdown)

    blocks = DualTrackMedicalParser(min_text_length=1)._track_a_text(
        Path("dummy.pdf"),
        "sha256",
    )

    assert len(blocks) == 1
    assert blocks[0].metadata.page_number == 4


def test_track_a_text_skips_reference_pages(monkeypatch):
    def fake_to_markdown(*args, **kwargs):
        return [
            {
                "metadata": {"page_number": 12},
                "text": "## ［参 考 文 献］\n［40］ Recommendations on off-label use of intravenous azithromycin in children［J］.",
            },
            {
                "metadata": {"page_number": 4},
                "text": "标准与讨论\n**==> picture [31 x 14] intentionally omitted <==**\nCAP：年龄3个月以上：10mg/kg 静脉滴注，qd，至少2天。",
            },
        ]

    monkeypatch.setattr(parser_module.pymupdf4llm, "to_markdown", fake_to_markdown)

    blocks = DualTrackMedicalParser(min_text_length=1)._track_a_text(
        Path("dummy.pdf"),
        "sha256",
    )

    assert len(blocks) == 1
    assert blocks[0].metadata.page_number == 4
    assert "参考文献" not in blocks[0].content
    assert "picture" not in blocks[0].content


def test_inspect_source_marks_scan_heavy_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def get_text(self):
            return ""

        def get_images(self, full=True):
            return [object()]

    class FakeDoc:
        page_count = 3

        def load_page(self, index):
            return FakePage()

        def close(self):
            pass

    monkeypatch.setattr(parser_module.fitz, "open", lambda _: FakeDoc())

    inspection = DualTrackMedicalParser().inspect_source(pdf_path, sample_pages=3)

    assert inspection.page_count == 3
    assert inspection.sampled_pages == 3
    assert inspection.text_pages == 0
    assert inspection.image_pages == 3
    assert inspection.scan_heavy is True


def test_serialize_evidence_chunks_uses_page_number():
    chunks = [
        SimpleNamespace(
            content="CAP：年龄3个月以上：10mg/kg 静脉滴注。",
            source_file="中国儿科超药品说明书用药专家共识.pdf",
            page_number=4,
        )
    ]

    payload = routes._serialize_evidence_chunks(chunks)

    assert payload == [
        {
            "content": "CAP：年龄3个月以上：10mg/kg 静脉滴注。",
            "source": "中国儿科超药品说明书用药专家共识.pdf",
            "page": 4,
        }
    ]


def test_resolve_answer_from_state_prefers_draft_answer():
    state = {"draft_answer": "依据不足，请人工复核。", "answer": "旧字段"}
    assert routes._resolve_answer_from_state(state) == "依据不足，请人工复核。"


def test_resolve_answer_from_state_falls_back_to_answer():
    state = {"answer": "旧字段回答"}
    assert routes._resolve_answer_from_state(state) == "旧字段回答"


def test_direct_prescription_request_is_blocked():
    assert routes._is_direct_prescription_request("这个孩子发热咳嗽 3 天，你帮我开处方。")
    assert routes._is_direct_prescription_request("患儿咳嗽发热，帮我开药。")
    assert not routes._is_direct_prescription_request("儿童重症肺炎支原体肺炎，是否可以静脉滴注阿奇霉素？")


def test_blocked_prescription_uses_rejected_trust_score():
    score = routes._rejected_trust_score()

    assert score.trust_level == TrustLevel.REJECTED
    assert score.trust_score == 0.0


def test_retriever_filters_reference_and_picture_noise():
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={"expert_consensus": 0.7, "default": 0.5}
    )

    response = {
        "documents": [[
            "## ［参 考 文 献］\n［40］ Recommendations on off-label use of intravenous azithromycin in children［J］.",
            "标准与讨论 **==> picture [31 x 14] intentionally omitted <==** CAP：年龄3个月以上：10mg/kg 静脉滴注。",
            "CAP：年龄3个月以上：10mg/kg 静脉滴注，qd，至少2天，然后5mg/kg口服。",
        ]],
        "metadatas": [[
            {"source_file": "中国儿科超药品说明书用药专家共识.pdf", "page_number": 12, "block_type": "text"},
            {"source_file": "中国儿科超药品说明书用药专家共识.pdf", "page_number": 4, "block_type": "text"},
            {"source_file": "中国儿科超药品说明书用药专家共识.pdf", "page_number": 4, "block_type": "text"},
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }

    chunks = retriever._parse_chroma_response(response, 128)

    assert len(chunks) == 1
    assert chunks[0].page_number == 4
    assert "CAP：年龄3个月以上" in chunks[0].content


def test_retriever_maps_new_guideline_sources_to_authority_weights():
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={
            "national_pharmacopoeia": 1.0,
            "clinical_guideline": 0.9,
            "expert_consensus": 0.7,
            "default": 0.5,
        }
    )

    assert retriever._get_authority_weight("儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf") == 0.9
    assert retriever._get_authority_weight("儿童社区获得性肺炎诊疗规范（2019年版）.pdf") == 0.9
    assert retriever._get_authority_weight("国家基本药物目录（2018年版）.pdf") == 1.0


def test_embedding_function_uses_dashscope_safe_batch_size():
    embed_fn = ZhipuEmbeddingFunction.__new__(ZhipuEmbeddingFunction)
    embed_fn._provider = "dashscope"

    assert embed_fn._batch_size() == 10


def test_context_intent_uses_multi_granularity_retrieval(monkeypatch):
    calls = []

    class FakeRetriever:
        def retrieve(self, query, granularity=None):
            calls.append({"query": query, "granularity": granularity})
            return []

    monkeypatch.setattr(retriever_node_module, "_retriever", FakeRetriever())

    state = {
        "original_query": "儿童重症肺炎支原体肺炎，是否可以静脉滴注阿奇霉素？",
        "normalized_query": "儿童 重症 肺炎支原体肺炎 静脉注射 阿奇霉素",
        "intent": IntentType.CONTEXT,
    }

    retriever_node_module.retriever_node(state)

    assert calls == [
        {
            "query": "儿童 重症 肺炎支原体肺炎 静脉注射 阿奇霉素",
            "granularity": None,
        }
    ]


def test_retriever_returns_empty_when_index_status_is_not_ready():
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._index_status = {
        "ready": False,
        "missing_sources": ["临床诊疗指南：小儿内科分册.pdf"],
    }

    assert retriever.retrieve("儿童阿奇霉素静脉滴注") == []


def test_retriever_filters_results_when_required_drug_term_is_absent():
    chunks = [
        SimpleNamespace(content="儿童支气管肺炎可根据需要进行退热、祛痰等对症治疗。"),
        SimpleNamespace(content="不推荐常规使用糖皮质激素。"),
    ]

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童支气管肺炎 氨溴索 超说明书 静脉给药",
        chunks,
    )

    assert filtered == []


def test_retriever_keeps_results_when_required_drug_alias_is_present():
    chunks = [
        SimpleNamespace(content="氨溴索可静脉给药。"),
        SimpleNamespace(content="不相关片段。"),
    ]

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童支气管肺炎 沐舒坦 静脉给药",
        chunks,
    )

    assert filtered == chunks[:1]


def test_retriever_requires_route_term_when_query_names_intravenous_use():
    chunks = [
        SimpleNamespace(content="阿奇霉素可用于肺炎支原体肺炎治疗。"),
        SimpleNamespace(content="重症推荐阿奇霉素静点，10mg/(kg.d)，qd。"),
    ]

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童重症肺炎支原体肺炎 静脉滴注 阿奇霉素",
        chunks,
    )

    assert filtered == chunks[1:]


def test_faithfulness_score_accepts_reasoning_alias():
    score = FaithfulnessScore.model_validate(
        {
            "score": 8,
            "reasoning": "The answer is supported by the evidence.",
        }
    )

    assert score.reason == "The answer is supported by the evidence."


def test_retriever_treats_missing_index_status_as_not_ready(tmp_path):
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._persist_dir = str(tmp_path)

    status = retriever._load_index_status()

    assert status["ready"] is False
    assert "index_status.json" in status["reason"]


def test_index_status_treats_scan_heavy_sources_as_incomplete():
    pdfs = [Path("text.pdf"), Path("scan.pdf")]
    per_doc_summary = {
        "text.pdf": {"blocks_total": 10},
        "scan.pdf": {"blocks_total": 1},
    }
    source_inspections = {
        "text.pdf": {"scan_heavy": False},
        "scan.pdf": {"scan_heavy": True},
    }

    status = _build_index_status(pdfs, per_doc_summary, source_inspections)

    assert status["ready"] is False
    assert status["indexed_sources"] == ["text.pdf", "scan.pdf"]
    assert status["incomplete_sources"] == ["scan.pdf"]


def test_settings_accepts_release_debug_values():
    settings = Settings.model_validate(
        {
            "DEBUG": "release",
            "LLM_PROVIDER": "zhipu",
            "EMBEDDING_PROVIDER": "zhipu",
        }
    )
    assert settings.DEBUG is False


def test_settings_env_file_points_to_backend_env():
    env_file = Path(Settings.model_config["env_file"])
    assert env_file.name == ".env"
    assert env_file.parent.name == "backend"


def test_settings_default_chroma_dir_points_to_backend_data():
    settings = Settings.model_validate(
        {
            "DEBUG": "release",
            "LLM_PROVIDER": "zhipu",
            "EMBEDDING_PROVIDER": "zhipu",
        }
    )
    chroma_dir = Path(settings.CHROMA_PERSIST_DIR)

    assert chroma_dir.parts[-3:] == ("backend", "data", "chroma_db")
