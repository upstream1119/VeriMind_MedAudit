"""
VeriMind-Med Trust-Score 计算器
=================================
实现论文 Eq.4 的门控数学公式:

    T = (α × S_ret + β × S_faith) × W_authority

其中:
    S_ret:      检索相关性得分 (0-10, 由 retriever 余弦相似度校准而来)
    S_faith:    忠实度得分 (0-10, 由 Judge LLM 评分)
    W_authority: 文献权威度权重 (0-1, 来自 config.AUTHORITY_WEIGHTS)
    α = 0.4, β = 0.6 (论文 Grid Search 最优解)

门控阈值 (config.TRUST_THRESHOLD_*):
    T ≥ τ_high (7.5) → TRUSTED
    τ_low (5.0) < T < τ_high → WARNING
    T ≤ τ_low (5.0) → REJECTED
"""

from __future__ import annotations

from app.config import get_settings
from app.models.schemas import TrustLevel, TrustScoreDetail


def compute_trust_score(
    s_ret: float,
    s_faith: float,
    w_authority: float,
    alpha: float | None = None,
    beta: float | None = None,
) -> TrustScoreDetail:
    """
    计算 Trust-Score 并返回完整的分项明细

    Args:
        s_ret:       检索相关性得分 [0, 10]
        s_faith:     忠实度得分 [0, 10]
        w_authority: 文献权威度权重 [0, 1]
        alpha:       α 权重 (默认读 config, 覆盖用于测试)
        beta:        β 权重 (默认读 config, 覆盖用于测试)

    Returns:
        TrustScoreDetail: 含完整分项明细和门控结果
    """
    settings = get_settings()
    a = alpha if alpha is not None else settings.TRUST_ALPHA
    b = beta if beta is not None else settings.TRUST_BETA

    # 输入有效性钳制 (防止异常值破坏计算)
    s_ret = max(0.0, min(10.0, s_ret))
    s_faith = max(0.0, min(10.0, s_faith))
    w_authority = max(0.0, min(1.0, w_authority))

    # Eq.4: T = (α × S_ret + β × S_faith) × W_authority
    trust_score = (a * s_ret + b * s_faith) * w_authority

    # 门控判定
    trust_level = _determine_trust_level(
        trust_score,
        tau_high=settings.TRUST_THRESHOLD_HIGH,
        tau_low=settings.TRUST_THRESHOLD_LOW,
    )

    return TrustScoreDetail(
        s_ret=round(s_ret, 4),
        s_faith=round(s_faith, 4),
        w_authority=round(w_authority, 4),
        trust_score=round(trust_score, 4),
        trust_level=trust_level,
        alpha=a,
        beta=b,
    )


def _determine_trust_level(
    score: float,
    tau_high: float,
    tau_low: float,
) -> TrustLevel:
    """根据阈值判定门控结果"""
    if score >= tau_high:
        return TrustLevel.TRUSTED
    elif score > tau_low:
        return TrustLevel.WARNING
    else:
        return TrustLevel.REJECTED


def relevance_to_score(distance: float, scale: float = 10.0) -> float:
    """
    将 ChromaDB L2 距离转换为 [0, 10] 的相关性得分

    公式: S_ret = scale / (1 + distance)

    Args:
        distance: ChromaDB 返回的 L2 距离 (越小越相关)
        scale:    得分缩放系数 (默认 10, 使结果在 0-10 范围)

    Returns:
        float: 相关性得分 [0, 10]
    """
    return scale / (1.0 + distance)
