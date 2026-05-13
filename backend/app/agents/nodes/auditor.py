"""
VeriMind-Med 智能体节点: Auditor (幻觉评分与门控裁判)
======================================================
职责:
  1. 获取 Generator 的 `draft_answer` 和 Retriever 提供给的 `evidence`
  2. 让高水平大模型 (Judge) 对草稿进行 "Faithfulness" (忠实度) 交叉审核打分 (0-10)
  3. 提取特征输入 `trust_score.py` 的物理数学公式
  4. 产出最终 Trust-Score 详情, 设置门控拦截(Reject / Warn)
"""

import logging
from pydantic import AliasChoices, BaseModel, Field
from app.agents.state import AuditState
from app.services.llm_client import generate_structured_output
from app.services.trust_score import compute_trust_score

logger = logging.getLogger(__name__)

class FaithfulnessScore(BaseModel):
    score: float = Field(..., ge=0, le=10, description="对源文档的忠实度打分，0 表示完全无中生有产生幻觉，10 表示严丝合缝")
    reason: str = Field(
        ...,
        validation_alias=AliasChoices("reason", "reasoning"),
        description="打分理由，简要说明判断依据",
    )

AUDITOR_SYSTEM_PROMPT = """你是一个冷酷、严厉的医疗大模型审计裁判 (Judge)。
你的唯一任务是判断【AI生成的草稿解答】是否 100% 忠实于【源文献片段】。

评分逻辑 (0-10分):
- 10分: 完美。解答完全来自源文献。特殊情况：如果草稿明确指出了“文献中未提及/依据不足”，这是对【文献无此信息】最忠实的反映，必须给满分 10分！
- 7-9分: 优秀。有极少量润色，或附加了必要的常规安全免责声明（如提示评估肝肾功能/过敏史等），且核心剂量、禁忌与文献完全吻合。
- 4-6分: 危险。有部分内容超出了文献范围，或者对数字（如剂量、年龄）有含糊、篡改的情况。
- 0-3分: 致命幻觉。只要文献中根本没提，但它装模作样编造了肯定的用药答复或具体剂量，必须给 0 分。

即使生成内容在现实中"可能是常识", 只要是对核心医学决策的盲目填充, 只要源片段里没有, 就必须扣分。

[源文献片段]:
{context}

[AI生成的草稿解答]:
{draft_answer}
"""

def auditor_node(state: AuditState) -> AuditState:
    """
    LangGraph Node: 审计与门控判决
    """
    draft_answer = state.get("draft_answer")
    evidence_chunks = state.get("evidence", [])
    
    logger.info("[Agent::Auditor] 启动裁判审计程序")
    
    if not draft_answer or not evidence_chunks:
        logger.warning("[Agent::Auditor] 资料不足, 判强制 REJECT")
        state["trust_score"] = compute_trust_score(0, 0, 0)
        state["current_node"] = "auditor"
        return state
        
    # 拼装裁判所看的上下文
    context_str = "\n\n".join([f"片段{i+1}: " + c.content for i, c in enumerate(evidence_chunks)])
    user_prompt = AUDITOR_SYSTEM_PROMPT.format(context=context_str, draft_answer=draft_answer)
    
    try:
        judgment: FaithfulnessScore = generate_structured_output(
            system_prompt="必须严格按照指定的 JSON Scheme 输出，并且【绝对禁止包含任何 Markdown 格式包裹 (如 ```json 或 ```)】，只允许输出单纯的 JSON 根节点。",
            user_prompt=user_prompt,
            output_schema=FaithfulnessScore,
            role="judge"
        )
        
        s_faith = judgment.score
        
        # 提取相关性 S_ret 和 权威度 w_authority (以召回列表中分数最高的第一条为主)
        top_chunk = evidence_chunks[0]
        s_ret = top_chunk.relevance_score * 10  # Schema 中要求是 [0,10] 的标度 (假设 relevance 已经是 [0,1])
        w_authority = top_chunk.authority_weight
        
        # 调用此前写的公式计算最终 Trust-Score
        ts_detail = compute_trust_score(s_ret=s_ret, s_faith=s_faith, w_authority=w_authority)
        
        state["trust_score"] = ts_detail
        state["current_node"] = "auditor"
        
        logger.info(f"[Agent::Auditor] 判决完成 | 忠实度得分:{s_faith} | 最终Trust-Score:{ts_detail.trust_score} | 结果:{ts_detail.trust_level.value}")
        logger.debug(f"[Agent::Auditor] 裁判说理: {judgment.reason}")
        
    except Exception as e:
        logger.error(f"[Agent::Auditor] 审计打分失败: {e}")
        state["error_message"] = f"Auditor 崩溃: {str(e)}"
        
    return state
