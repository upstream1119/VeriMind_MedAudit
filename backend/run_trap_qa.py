"""
VeriMind-Med 批量 Trap-QA 系统回答记录生成器
=============================================
遍历 tests/trap_qa.json 中所有陷阱题，
依次喂给 Agent 引擎，将输出保存到
  docs/test_runs/YYYY-MM-DD_HH-MM/
每题一个 Markdown 文件，同时生成一份汇总文件
"""
import sys, asyncio, json, logging, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

# ── 目录定位 ──
SCRIPT_DIR = Path(__file__).parent            # backend/
PROJECT_ROOT = SCRIPT_DIR.parent             # VeriMind_MedAudit/

RUN_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUT_DIR  = PROJECT_ROOT / "docs" / "test_runs" / RUN_TIME
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    filename=str(OUT_DIR / "batch_log.txt"),
    filemode="w",
    encoding="utf-8"
)

from app.agents.state import AuditState
from app.agents.graph import audit_engine

# ── 加载题库 ──
TRAP_QA_PATH = PROJECT_ROOT / "backend" / "tests" / "trap_qa.json"

def load_questions():
    with open(TRAP_QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]

def make_md(case: dict, final_state: dict) -> str:
    """把单题的完整运行结果写成 Markdown 字符串"""
    ts = final_state.get("trust_score")
    trust_level = ts.trust_level.value if ts else "未执行"
    trust_score  = f"{ts.trust_score:.4f}" if ts else "N/A"
    s_ret        = f"{ts.s_ret:.4f}"   if ts else "N/A"
    s_faith      = f"{ts.s_faith:.4f}" if ts else "N/A"
    w_auth       = f"{ts.w_authority:.4f}" if ts else "N/A"

    level_icon = {"TRUSTED": "✅", "WARNING": "⚠️", "REJECTED": "🚫"}.get(trust_level, "❓")

    lines = [
        f"# {case['id']} — {case['trap_type']}（难度：{case['difficulty']}）",
        "",
        f"> **陷阱描述：** {case['trap_description']}",
        "",
        "---",
        "",
        "## 用户提问",
        "",
        f"> {case['question']}",
        "",
        "---",
        "",
        "## 系统回答",
        "",
        "### 知识对齐层",
        "",
        f"- **标准化查询：** `{final_state.get('normalized_query') or '（对齐失败，退化为原始问句）'}`",
        f"- **识别意图：** `{final_state.get('intent') or 'N/A'}`",
        f"- **召回证据数：** {len(final_state.get('evidence', []))} 条",
        "",
        "### 推理生成层（Generator 回答）",
        "",
        final_state.get("draft_answer") or "_（生成失败）_",
        "",
        "---",
        "",
        "## 审计门控结果",
        "",
        f"| 指标 | 值 |",
        f"|:---|:---|",
        f"| **门控判定** | {level_icon} **{trust_level}** |",
        f"| Trust-Score | {trust_score} / 10 |",
        f"| 相关性 S_ret | {s_ret} |",
        f"| 忠实度 S_faith | {s_faith} |",
        f"| 权威度 W | {w_auth} |",
        "",
        "---",
        "",
        "## 正确答案参考",
        "",
        f"**正确关键词：** {', '.join(case['correct_keywords'])}",
        "",
        f"**预期门控等级：** `{case['expected_trust_level']}`",
        "",
        f"**临床备注：** {case['clinical_note']}",
        "",
        "---",
        "",
        "## 临床专家评审栏",
        "",
        "| 维度 | 评分（1-5） | 评语 |",
        "|:---|:---:|:---|",
        "| 核心结论是否正确？ | ___ | |",
        "| 警示是否符合临床？ | ___ | |",
        "| 整体回答是否可信赖？ | ___ | |",
        "",
        "**综合评价：** [ 完全认可 ] / [ 基本认可 ] / [ 有明显问题 ]",
        "",
        "**文字评语（可选）：**",
        "",
        "> ___________________________________________",
        "",
        f"_记录时间：{RUN_TIME}_",
    ]
    return "\n".join(lines)

async def run_single(case: dict) -> dict:
    """对单道题跑完整管线，返回 final_state"""
    initial = AuditState(
        original_query=case["question"],
        normalized_query=None,
        intent=None,
        evidence=[],
        draft_answer=None,
        trust_score=None,
        current_node="START",
        error_message=None
    )
    return await audit_engine.ainvoke(initial)

async def main():
    cases = load_questions()
    summary_rows = []

    # 仅运行第 3 题 (TQ-003) 补充数据
    START_OFFSET = 2
    cases_test = cases[START_OFFSET : START_OFFSET+1]

    for i, case in enumerate(cases_test, start=START_OFFSET):
        print(f"[{i+1}/{len(cases)}] 正在测试 {case['id']}: {case['question'][:30]}...")
        try:
            final = await run_single(case)
        except Exception as e:
            print(f"  ❌ 执行失败: {e}")
            continue

        # 写单题文件
        md_text = make_md(case, final)
        out_file = OUT_DIR / f"{case['id']}.md"
        out_file.write_text(md_text, encoding="utf-8")
        print(f"  ✅ 已存入: {out_file}")

        # 避免触发智谱免费接口并发限流，休眠5秒
        print(f"  ⏳ 冷却 5 秒防止 API 限流...")
        await asyncio.sleep(5)

        # 整理汇总行
        ts = final.get("trust_score")
        actual_level = ts.trust_level.value if ts else "ERROR"
        expected     = case["expected_trust_level"]
        match = "✅" if actual_level == expected else "❌"
        summary_rows.append(
            f"| {case['id']} | {case['trap_type']} | {case['difficulty']} | "
            f"{actual_level} | {expected} | {match} | "
            f"{f'{ts.trust_score:.2f}' if ts else 'N/A'} |"
        )

    # 写汇总文件
    summary_md = "\n".join([
        "# VeriMind-Med Trap-QA 批量测试汇总报告",
        "",
        f"运行时间：`{RUN_TIME}`  |  题目总数：{len(cases)}",
        "",
        "| 题号 | 陷阱类型 | 难度 | 实际判定 | 预期判定 | 是否符合 | Trust-Score |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
        *summary_rows,
        "",
        "---",
        "",
        "## 各题详细报告",
        "",
        *[f"- [{c['id']}](./{c['id']}.md)（{c['trap_type']}）" for c in cases],
    ])
    summary_file = OUT_DIR / "00_SUMMARY.md"
    summary_file.write_text(summary_md, encoding="utf-8")
    print(f"\n📋 汇总报告已生成: {summary_file}")

if __name__ == "__main__":
    asyncio.run(main())
