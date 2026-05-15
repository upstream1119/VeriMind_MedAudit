"""
VeriMind-Med 智能体节点: Router & Alignment (导诊与知识对齐台)
===========================================================
职责:
  1. 接收用户的口语化提问
  2. 提取标准医学术语，消除歧义，纠正白话 (如: 退烧药 -> 对乙酰氨基酚/布洛芬)
  3. 判定该查询最适合的颗粒度意图 (DETAIL: 128 / CONCEPT: 512 / CONTEXT: 1024)
"""

import logging
from pydantic import BaseModel, Field
from app.models.schemas import IntentType
from app.agents.state import AuditState
from app.services.llm_client import generate_structured_output
from app.services.trust_score import compute_trust_score

logger = logging.getLogger(__name__)

# 定义 Router 的结构化输出格式
class RouterDecision(BaseModel):
    normalized_query: str = Field(
        ..., 
        description="对齐后的专业医学检索短句，用于在向量库中执行高精度匹配。需包含核心药物和病症。"
    )
    intent: IntentType = Field(
        ..., 
        description="查询意图。DETAIL(问具体精确剂量、禁忌等), CONCEPT(问某药理机制、适应群体), CONTEXT(复杂的病案多药分析)"
    )


ROUTER_SYSTEM_PROMPT = """你是一个专业的儿科临床医学文献检索重写专家。
用户的提问通常是口语化的，你需要：
1. 知识对齐：将用户的口语表述转化为标准的医学与药学通用名或术语集合。例如“小孩发烧39度吃扑热息痛”应转化为“儿童 高热 对乙酰氨基酚”。
2. 意图判别：判断该问题在阅读文献时，需要何种粒度的知识碎片：
   - DETAIL: 适合找具体数字（如剂量、年龄界限、肾功能指标）。
   - CONCEPT: 适合找某个机制原理、适应症大段描述。
   - CONTEXT: 适合找全章节阅读，如对比治疗方案、复杂并发表征。

   - CONTEXT: 适合找全章节阅读，如对比治疗方案、复杂并发表征。

必须严格按照指定的 JSON Scheme 输出，并且【绝对禁止包含任何 Markdown 格式包裹 (如 ```json 或 ```)】，只允许输出单纯的 JSON 根节点。
【系统级严重警告】：返回的 JSON 中，必须并且只能使用 "normalized_query" 和 "intent" 这两个键名！禁止擅自编造如 knowledge_alignment 等其他名字！
"""


def _reject_router_failure(state: AuditState, original_query: str, error: str) -> AuditState:
    state["normalized_query"] = original_query
    state["intent"] = IntentType.DETAIL
    state["evidence"] = []
    state["draft_answer"] = (
        "已拦截：问题路由解析失败，无法可靠识别医学审计意图。"
        "请补充更明确的药物、剂量、给药途径，或交由人工复核。"
    )
    state["trust_score"] = compute_trust_score(0, 0, 0)
    state["current_node"] = "router"
    state["error_message"] = f"Router 解析失败: {error}"
    return state


def router_node(state: AuditState) -> AuditState:
    """
    LangGraph Node: 意图路由与医学术语提取
    """
    original_query = state["original_query"]
    logger.info(f"[Agent::Router] 开始分析临床查询: {original_query}")
    
    try:
        # 调用 Router LLM 执行结构化解析
        decision: RouterDecision = generate_structured_output(
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_prompt=original_query,
            output_schema=RouterDecision,
            role="router"
        )
        if not decision.normalized_query.strip():
            raise ValueError("Router returned empty normalized_query")
        
        # 更新图状态
        state["normalized_query"] = decision.normalized_query
        state["intent"] = decision.intent
        state["current_node"] = "router"
        
        logger.info(f"[Agent::Router] 决策结果 | 意图:{decision.intent.value} | 对齐语句:{decision.normalized_query}")
        
    except Exception as e:
        logger.error(f"[Agent::Router] 结构化解析失败: {e}")
        return _reject_router_failure(state, original_query, str(e))
        
    return state
