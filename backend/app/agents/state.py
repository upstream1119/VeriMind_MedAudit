"""
VeriMind-Med 智能体状态 (Graph State)
=====================================
这是 LangGraph 在各个节点之间传递的唯一数据结构 (State)
包含了用户请求的原始数据以及中间链路的分析结果
"""
from typing import TypedDict, Optional, List
from app.models.schemas import IntentType, RetrievedChunk, TrustScoreDetail

class AuditState(TypedDict):
    """
    审计过程的有向图状态定义
    使用 TypedDict 配合 LangGraph 要求
    """
    # ── 1. 输入阶段 ──
    original_query: str                  # 用户的原始提问
    
    # ── 2. 路由与对齐阶段 ──
    normalized_query: Optional[str]      # 经过医学术语规范化后的查询词
    intent: Optional[IntentType]         # 查询意图 (DETAIL / CONCEPT / CONTEXT)
    
    # ── 3. 检索阶段 ──
    evidence: List[RetrievedChunk]       # 检索返回的权威依据片段
    
    # ── 4. 推理阶段 ──
    draft_answer: Optional[str]          # Generator 大模型基于 evidence 起草的回答
    
    # ── 5. 审计阶段 ──
    trust_score: Optional[TrustScoreDetail]  # Auditor 大模型计算并综合得出的 Trust-Score 详情
    
    # ── 6. 状态流转标识 ──
    current_node: str                    # 当前执行节点标识
    error_message: Optional[str]         # 如果发生阻断或故障, 记录说明
