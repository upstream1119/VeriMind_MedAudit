"""
运行人工指定的高优先级问答回归，并将结果写入 docs/test_runs。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.agents.graph import audit_engine
from app.api.routes import _resolve_answer_from_state, _serialize_evidence_chunks


CASES = [
    {
        "id": "MQ-001",
        "question": "儿童重症肺炎支原体肺炎，是否可以静脉滴注阿奇霉素？有什么依据？",
    },
    {
        "id": "MQ-002",
        "question": "小儿支气管肺炎，能否超说明书静脉使用沐舒坦（氨溴索）？",
    },
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _output_dir() -> Path:
    root = _project_root()
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = root / "docs" / "test_runs" / run_time
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_index_status(root: Path) -> dict[str, object]:
    status_path = root / "backend" / "data" / "chroma_db" / "index_status.json"
    if not status_path.exists():
        return {
            "ready": False,
            "reason": f"{status_path.name} missing",
        }

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"failed to read {status_path.name}: {exc}",
        }

    if not isinstance(status, dict):
        return {
            "ready": False,
            "reason": f"{status_path.name} is not a JSON object",
        }
    status.setdefault("ready", False)
    return status


def _render_case_markdown(case: dict, state: dict) -> str:
    trust = state.get("trust_score")
    evidence = _serialize_evidence_chunks(state.get("evidence", []))
    answer = _resolve_answer_from_state(state)

    lines = [
        f"# {case['id']}",
        "",
        f"**问题：** {case['question']}",
        "",
        f"**标准化查询：** {state.get('normalized_query', 'N/A')}",
        f"**意图：** {state.get('intent', 'N/A')}",
        f"**回答：** {answer}",
        "",
        "## Trust-Score",
        "",
        f"- trust_level: {getattr(trust, 'trust_level', 'N/A')}",
        f"- trust_score: {getattr(trust, 'trust_score', 'N/A')}",
        f"- s_ret: {getattr(trust, 's_ret', 'N/A')}",
        f"- s_faith: {getattr(trust, 's_faith', 'N/A')}",
        f"- w_authority: {getattr(trust, 'w_authority', 'N/A')}",
        "",
        "## Evidence",
        "",
    ]

    if not evidence:
        lines.append("- 无证据返回")
    else:
        for idx, ev in enumerate(evidence, start=1):
            lines.extend(
                [
                    f"### 依据 {idx}",
                    f"- 来源：{ev['source']}",
                    f"- 页码：{ev['page']}",
                    f"- 内容：{ev['content']}",
                    "",
                ]
            )

    return "\n".join(lines)


async def _run_case(case: dict) -> dict:
    return await audit_engine.ainvoke({"original_query": case["question"]})


async def main() -> None:
    root = _project_root()
    out_dir = _output_dir()
    index_status = _load_index_status(root)
    if not index_status.get("ready", False):
        blocked = {
            "status": "blocked",
            "reason": index_status.get("reason") or "index is not ready",
            "missing_sources": index_status.get("missing_sources", []),
            "scan_heavy_sources": index_status.get("scan_heavy_sources", []),
        }
        (out_dir / "00_SUMMARY.json").write_text(
            json.dumps(blocked, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "00_BLOCKED.md").write_text(
            "\n".join([
                "# 人工回归已阻塞",
                "",
                f"原因：{blocked['reason']}",
                f"缺失资料：{blocked['missing_sources']}",
                f"扫描件倾向资料：{blocked['scan_heavy_sources']}",
            ]),
            encoding="utf-8",
        )
        print(f"索引未就绪，人工回归已阻塞: {out_dir}")
        return

    summary = []

    for case in CASES:
        state = await _run_case(case)
        md = _render_case_markdown(case, state)
        (out_dir / f"{case['id']}.md").write_text(md, encoding="utf-8")
        summary.append(
            {
                "id": case["id"],
                "question": case["question"],
                "normalized_query": state.get("normalized_query"),
                "intent": str(state.get("intent")),
                "answer": _resolve_answer_from_state(state),
                "trust_level": str(getattr(state.get("trust_score"), "trust_level", "N/A")),
                "evidence_count": len(state.get("evidence", [])),
            }
        )

    (out_dir / "00_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"人工回归结果已写入: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
