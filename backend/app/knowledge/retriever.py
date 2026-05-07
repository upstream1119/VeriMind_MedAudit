"""
VeriMind-Med 多粒度检索服务层 (Multi-Granularity Retriever)
=============================================================
职责:
  1. 封装三粒度 ChromaDB 查询, 暴露统一的 retrieve() 接口
  2. 对检索结果进行权威度加权 (W_authority), 优先返回权威来源切片
  3. 支持单粒度查询和三粒度融合查询两种模式

权威度权重规则 (来自 config.AUTHORITY_WEIGHTS):
  国家药典 (1.0) > 临床诊疗指南 (0.9) > 专家共识 (0.7)
    > 教材 (0.6) > 未标注 (0.5) > 个案报道 (0.3)

文件名 → 权威等级映射 (基于我们已知的三本书目):
  《国家基本药物处方集》  → national_pharmacopoeia (1.0)
  《临床诊疗指南·小儿内科分册》 → clinical_guideline (0.9)
  《儿科超说明书用药专家共识》  → expert_consensus (0.7)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import chromadb

from app.config import get_settings
from app.knowledge.indexer import ZhipuEmbeddingFunction, _COLLECTION_NAMES

logger = logging.getLogger(__name__)

# 文件名关键词 → 权威等级
_AUTHORITY_KEYWORD_MAP = {
    "处方集": "national_pharmacopoeia",
    "basic_drug": "national_pharmacopoeia",
    "诊疗指南": "clinical_guideline",
    "内科分册": "clinical_guideline",
    "guideline": "clinical_guideline",
    "超说明书": "expert_consensus",
    "专家共识": "expert_consensus",
    "consensus": "expert_consensus",
}


# ────────────────────────────────────────────
# 返回数据结构
# ────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """检索结果单元, 带完整溯源信息和权威度评分"""
    content: str              # 切片文本
    granularity: int          # 所属粒度 (128/512/1024)
    distance: float           # 向量距离 (越小越相关, ChromaDB 默认 L2)
    relevance_score: float    # 相关性分数 = 1 / (1 + distance)
    authority_weight: float   # 权威度权重 (来自 config)
    final_score: float        # 最终综合分 = relevance_score × authority_weight
    source_file: str          # 来源文件名
    page_number: int          # 来源页码
    chapter_title: str        # 来源章节
    block_type: str           # 块类型 (text/table)


# ────────────────────────────────────────────
# 多粒度检索器
# ────────────────────────────────────────────
class MultiGranularityRetriever:
    """
    三粒度 ChromaDB 检索器

    使用方法:
        retriever = MultiGranularityRetriever()

        # 三粒度融合检索
        results = retriever.retrieve("阿莫西林儿童剂量", top_k=5)

        # 单粒度检索
        results = retriever.retrieve("阿莫西林儿童剂量", top_k=5, granularity=128)
    """

    def __init__(self, persist_dir: str | None = None):
        settings = get_settings()
        self._settings = settings
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR

        # ChromaDB 只读客户端
        self._chroma = chromadb.PersistentClient(path=self._persist_dir)

        # Embedding 函数 (与 indexer 共享同一实现)
        self._embed_fn = ZhipuEmbeddingFunction()

        logger.info(f"[Retriever] 初始化完成: {self._persist_dir}")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        granularity: Literal[128, 512, 1024] | None = None,
        include_tables: bool = True,
    ) -> list[RetrievedChunk]:
        """
        执行多粒度语义检索

        Args:
            query: 查询文本 (儿科用药相关问题)
            top_k: 每个粒度返回的结果数, 默认读 config.RETRIEVAL_TOP_K
            granularity: 指定单粒度检索; None = 三粒度融合
            include_tables: 是否包含表格类型的切片

        Returns:
            按 final_score 降序排列的检索结果列表
        """
        k = top_k or self._settings.RETRIEVAL_TOP_K

        # 生成查询向量
        logger.info(f"[Retriever] 查询: '{query[:50]}...' top_k={k}")
        query_embedding = self._embed_fn([query])[0]

        # 确定检索哪些粒度
        if granularity is not None:
            target_granularities = [granularity]
        else:
            target_granularities = [128, 512, 1024]

        all_results: list[RetrievedChunk] = []

        for g in target_granularities:
            col_name = _COLLECTION_NAMES.get(g)
            if col_name is None:
                continue
            try:
                collection = self._chroma.get_collection(name=col_name)
            except Exception:
                logger.warning(f"[Retriever] Collection {col_name} 不存在, 跳过")
                continue

            if collection.count() == 0:
                logger.warning(f"[Retriever] Collection {col_name} 为空, 跳过")
                continue

            # 构建 where 过滤条件 (排除表格)
            where = None
            if not include_tables:
                where = {"block_type": {"$eq": "text"}}

            # ChromaDB 向量查询
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            chunks = self._parse_chroma_response(response, g)
            all_results.extend(chunks)
            logger.info(f"[Retriever] 粒度 {g}: 命中 {len(chunks)} 条")

        # 权威度加权 + 排序
        for chunk in all_results:
            chunk.final_score = chunk.relevance_score * chunk.authority_weight

        all_results.sort(key=lambda c: c.final_score, reverse=True)

        logger.info(f"[Retriever] 检索完成, 共返回 {len(all_results)} 条结果")
        return all_results

    def get_stats(self) -> dict[str, int]:
        """获取各 Collection 的文档数量"""
        stats = {}
        for g, name in _COLLECTION_NAMES.items():
            try:
                col = self._chroma.get_collection(name=name)
                stats[name] = col.count()
            except Exception:
                stats[name] = 0
        return stats

    # ── 内部方法 ──

    def _parse_chroma_response(
        self, response: dict, granularity: int
    ) -> list[RetrievedChunk]:
        """解析 ChromaDB 返回结果, 注入权威度权重"""
        results = []

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # 相关性分数: 从 L2 距离转换, 距离越小分越高
            relevance = 1.0 / (1.0 + dist)

            # 权威度: 根据文件名关键词确定
            authority = self._get_authority_weight(meta.get("source_file", ""))

            results.append(RetrievedChunk(
                content=doc,
                granularity=granularity,
                distance=dist,
                relevance_score=relevance,
                authority_weight=authority,
                final_score=0.0,  # 在外层统一计算
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 0),
                chapter_title=meta.get("chapter_title", ""),
                block_type=meta.get("block_type", "text"),
            ))

        return results

    def _get_authority_weight(self, filename: str) -> float:
        """根据文件名关键词返回对应的权威度权重"""
        weights = self._settings.AUTHORITY_WEIGHTS
        for keyword, level in _AUTHORITY_KEYWORD_MAP.items():
            if keyword in filename:
                return weights.get(level, weights["default"])
        return weights["default"]
