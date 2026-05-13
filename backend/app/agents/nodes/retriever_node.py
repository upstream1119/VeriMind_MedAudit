"""
VeriMind-Med 智能体节点: Retriever (向量检索执行器)
======================================================
职责:
  1. 读取 Router 产出的 `normalized_query` 和 `intent`
  2. 根据意图决定去哪个粒度查询 (DETAIL->128, CONCEPT->512, CONTEXT->1024)
  3. 执行检索并将回传的 List[RetrievedChunk] 压入 State
"""

import logging
from app.agents.state import AuditState
from app.knowledge.retriever import MultiGranularityRetriever
from app.models.schemas import IntentType

logger = logging.getLogger(__name__)

# 全局单例
_retriever = MultiGranularityRetriever()

def retriever_node(state: AuditState) -> AuditState:
    """
    LangGraph Node: 向量检索执行
    """
    query = state.get("normalized_query") or state["original_query"]
    intent = state.get("intent", IntentType.CONCEPT)
    
    intent_val = intent.value if hasattr(intent, 'value') else str(intent)
    
    logger.info(f"[Agent::Retriever] 开始依据意图 {intent_val} 检索: {query}")
    
    # 意图映射到具体粒度 (如果是意图模糊的, 退化为三粒度融合)
    granularity = None
    if intent_val == IntentType.DETAIL.value or intent_val == "DETAIL":
        granularity = 128
    elif intent_val == IntentType.CONCEPT.value or intent_val == "CONCEPT":
        granularity = 512
    elif intent_val == IntentType.CONTEXT.value or intent_val == "CONTEXT":
        granularity = None
        
    try:
        # 调用此前阶段写好的检索层
        # top_k 默认走 config 中的配置
        evidence_chunks = _retriever.retrieve(
            query=query,
            granularity=granularity
        )
        
        state["evidence"] = evidence_chunks
        state["current_node"] = "retriever"
        
        logger.info(f"[Agent::Retriever] 检索成功, 共召回 {len(evidence_chunks)} 条权威片段")
    except Exception as e:
        logger.error(f"[Agent::Retriever] 向量检索失败: {e}")
        state["error_message"] = f"Retriever 崩溃: {str(e)}"
        state["evidence"] = []
        
    return state
