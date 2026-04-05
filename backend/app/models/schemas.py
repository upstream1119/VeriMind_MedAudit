"""
VeriMind-Med 数据模型 (Pydantic Schemas)
定义 API 请求/响应、Agent 流转中间态的数据结构
"""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# ── 枚举类型 ──

class TrustLevel(str, Enum):
    """Trust-Score 门控结果"""
    TRUSTED = "TRUSTED"       # T ≥ 7.5
    WARNING = "WARNING"       # 5.0 < T < 7.5
    REJECTED = "REJECTED"     # T ≤ 5.0


class IntentType(str, Enum):
    """查询意图类型 (论文 Algorithm 1)"""
    DETAIL = "DETAIL"         # 128 tokens — 精确参数、剂量
    CONCEPT = "CONCEPT"       # 512 tokens — 定义、方法论
    CONTEXT = "CONTEXT"       # 1024 tokens — 综合分析、多文档对比


class EvidenceLevel(str, Enum):
    """医学证据等级"""
    NATIONAL_PHARMACOPOEIA = "national_pharmacopoeia"   # 国家药典
    CLINICAL_GUIDELINE = "clinical_guideline"             # 临床诊疗指南
    EXPERT_CONSENSUS = "expert_consensus"                 # 专家共识
    TEXTBOOK = "textbook"                                 # 教材
    CASE_REPORT = "case_report"                           # 个案报道
    DEFAULT = "default"                                   # 未标注


# ── API 请求模型 ──

class AuditQueryRequest(BaseModel):
    """审计查询请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户提问")
    session_id: str | None = Field(default=None, description="会话 ID")


# ── Agent 流转中间态 ──

class RetrievedChunk(BaseModel):
    """检索到的文档片段"""
    content: str = Field(..., description="文本内容")
    source: str = Field(default="", description="来源文件名")
    page: int | None = Field(default=None, description="PDF 页码")
    chunk_id: str = Field(default="", description="片段唯一标识")
    similarity_score: float = Field(default=0.0, description="余弦相似度")
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.DEFAULT, description="证据等级"
    )
    authority_weight: float = Field(default=0.5, description="权威度权重 W_authority")


class TrustScoreDetail(BaseModel):
    """Trust-Score 详细分解"""
    s_ret: float = Field(..., ge=0, le=10, description="检索相关性得分 (校准后)")
    s_faith: float = Field(..., ge=0, le=10, description="忠实度得分")
    w_authority: float = Field(..., ge=0, le=1, description="证据权威度加权值")
    trust_score: float = Field(..., ge=0, le=10, description="最终 Trust-Score T")
    trust_level: TrustLevel = Field(..., description="门控结果")
    alpha: float = Field(default=0.4, description="α 权重")
    beta: float = Field(default=0.6, description="β 权重")


# ── API 响应模型 ──

class AuditQueryResponse(BaseModel):
    """审计查询响应"""
    query: str = Field(..., description="原始提问")
    normalized_query: str = Field(default="", description="对齐后的标准化查询")
    intent: IntentType = Field(..., description="识别到的查询意图")
    answer: str = Field(..., description="生成的回答")
    trust_score: TrustScoreDetail = Field(..., description="Trust-Score 详情")
    evidence: list[RetrievedChunk] = Field(
        default_factory=list, description="检索到的证据链"
    )
    processing_time: float = Field(default=0.0, description="处理耗时 (秒)")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    app_name: str = ""
    version: str = ""
    llm_provider: str = ""
    models: dict = Field(default_factory=dict)
