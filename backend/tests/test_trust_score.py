"""
Trust-Score 计算逻辑 Pytest 单元测试
======================================
覆盖范围:
  1. Trust-Score 数学公式 (Eq.4)
  2. 三档门控阈值判定 (TRUSTED / WARNING / REJECTED)
  3. 边界值与极端输入防御 (钳制逻辑)
  4. L2 距离 → 相关性得分转换
"""

import pytest

from app.services.trust_score import compute_trust_score, relevance_to_score
from app.models.schemas import TrustLevel


# ── 测试 Trust-Score 数学公式 ──

class TestTrustScoreFormula:
    """验证 T = (α × S_ret + β × S_faith) × W_authority"""

    def test_basic_formula(self):
        """标准参数计算, 验证公式正确"""
        # T = (0.4 × 8.0 + 0.6 × 9.0) × 0.9
        # T = (3.2 + 5.4) × 0.9 = 8.6 × 0.9 = 7.74
        result = compute_trust_score(s_ret=8.0, s_faith=9.0, w_authority=0.9)
        assert abs(result.trust_score - 7.74) < 0.01

    def test_alpha_beta_override(self):
        """自定义 α/β 时计算结果正确"""
        # T = (0.3 × 6.0 + 0.7 × 8.0) × 1.0 = (1.8 + 5.6) = 7.4
        result = compute_trust_score(
            s_ret=6.0, s_faith=8.0, w_authority=1.0,
            alpha=0.3, beta=0.7
        )
        assert abs(result.trust_score - 7.4) < 0.01

    def test_authority_weight_scales_score(self):
        """权威度权重正确地缩放最终得分"""
        result_high = compute_trust_score(s_ret=8.0, s_faith=8.0, w_authority=1.0)
        result_low  = compute_trust_score(s_ret=8.0, s_faith=8.0, w_authority=0.5)
        assert abs(result_high.trust_score - result_low.trust_score * 2) < 0.01

    def test_detail_fields_are_preserved(self):
        """返回的 TrustScoreDetail 包含完整分项数据"""
        result = compute_trust_score(s_ret=7.0, s_faith=6.0, w_authority=0.8)
        assert result.s_ret == 7.0
        assert result.s_faith == 6.0
        assert result.w_authority == 0.8
        assert result.alpha == 0.4  # config 默认值
        assert result.beta == 0.6


# ── 测试门控阈值判定 ──

class TestTrustLevelGating:
    """
    验证三档门控阈值 (τ_high=7.5, τ_low=5.0):
      T ≥ 7.5 → TRUSTED
      5.0 < T < 7.5 → WARNING
      T ≤ 5.0 → REJECTED
    """

    def test_trusted_threshold(self):
        """T = 7.5 (边界值) → TRUSTED"""
        # 构造刚好产生 T=7.5 的参数: (0.4×7.5 + 0.6×7.5) × 1.0 = 7.5
        result = compute_trust_score(s_ret=7.5, s_faith=7.5, w_authority=1.0)
        assert result.trust_level == TrustLevel.TRUSTED

    def test_above_trusted_threshold(self):
        """T 远高于 7.5 → TRUSTED"""
        result = compute_trust_score(s_ret=10.0, s_faith=10.0, w_authority=1.0)
        assert result.trust_level == TrustLevel.TRUSTED

    def test_warning_zone(self):
        """T 在 5.0 到 7.5 之间 → WARNING"""
        # T = (0.4×6.0 + 0.6×7.0) × 1.0 = (2.4 + 4.2) = 6.6
        result = compute_trust_score(s_ret=6.0, s_faith=7.0, w_authority=1.0)
        assert result.trust_level == TrustLevel.WARNING
        assert 5.0 < result.trust_score < 7.5

    def test_rejected_threshold(self):
        """T = 5.0 (边界值) → REJECTED"""
        # T = (0.4×5.0 + 0.6×5.0) × 1.0 = 5.0
        result = compute_trust_score(s_ret=5.0, s_faith=5.0, w_authority=1.0)
        assert result.trust_level == TrustLevel.REJECTED

    def test_below_rejected_threshold(self):
        """T 远低于 5.0 → REJECTED"""
        result = compute_trust_score(s_ret=1.0, s_faith=1.0, w_authority=0.3)
        assert result.trust_level == TrustLevel.REJECTED

    def test_low_authority_demotes_to_rejected(self):
        """高原始分但低权威度 → 可能降为 REJECTED"""
        # T = (0.4×8.0 + 0.6×8.0) × 0.5 = 8.0 × 0.5 = 4.0 → REJECTED
        result = compute_trust_score(s_ret=8.0, s_faith=8.0, w_authority=0.5)
        assert result.trust_level == TrustLevel.REJECTED


# ── 测试边界值与防御性输入 ──

class TestInputClamping:
    """验证超出范围的输入被安全钳制"""

    def test_scores_clamped_at_max(self):
        """超过 10 的输入被钳制到 10"""
        result = compute_trust_score(s_ret=15.0, s_faith=20.0, w_authority=1.0)
        assert result.s_ret == 10.0
        assert result.s_faith == 10.0

    def test_scores_clamped_at_min(self):
        """低于 0 的输入被钳制到 0"""
        result = compute_trust_score(s_ret=-5.0, s_faith=-3.0, w_authority=0.5)
        assert result.s_ret == 0.0
        assert result.s_faith == 0.0

    def test_authority_weight_clamped(self):
        """权威度权重超出 [0,1] 范围时被钳制"""
        result_over  = compute_trust_score(s_ret=8.0, s_faith=8.0, w_authority=2.0)
        result_under = compute_trust_score(s_ret=8.0, s_faith=8.0, w_authority=-1.0)
        assert result_over.w_authority == 1.0
        assert result_under.w_authority == 0.0

    def test_zero_authority_yields_zero_score(self):
        """权威度为 0 时, Trust-Score 必为 0"""
        result = compute_trust_score(s_ret=10.0, s_faith=10.0, w_authority=0.0)
        assert result.trust_score == 0.0
        assert result.trust_level == TrustLevel.REJECTED


# ── 测试 L2 距离 → 相关性得分转换 ──

class TestRelevanceConversion:
    """验证 S_ret = 10 / (1 + distance) 的转换逻辑"""

    def test_zero_distance_yields_max_score(self):
        """distance = 0 (完全匹配) → S_ret = 10.0"""
        assert relevance_to_score(0.0) == 10.0

    def test_score_decreases_with_distance(self):
        """距离越大, 相关性得分越低"""
        assert relevance_to_score(0.1) > relevance_to_score(0.5)
        assert relevance_to_score(0.5) > relevance_to_score(1.0)

    def test_large_distance_approaches_zero(self):
        """极大距离时得分趋近 0"""
        score = relevance_to_score(999.0)
        assert score < 0.02

    def test_typical_chroma_distance(self):
        """典型 ChromaDB L2 距离 (~0.3) 的得分在合理范围"""
        # S_ret = 10 / (1 + 0.3) ≈ 7.69
        score = relevance_to_score(0.3)
        assert 7.0 < score < 8.5
