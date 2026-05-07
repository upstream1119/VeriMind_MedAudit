"""
VeriMind-Med 动态语义切分引擎 (Semantic Chunker)
==================================================
核心设计原则:
  1. 绝不在句子中间截断 —— 以"。；;.!！?？"为绝对硬边界
  2. 每个切片必须继承父块的全部溯源元数据
  3. 支持三粒度切分: 128(细节) / 512(概念) / 1024(上下文) tokens
  4. 表格块整体保留, 不做二次切分 (防止剂量表格错乱)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from app.knowledge.parser import ParsedBlock, BlockMetadata

logger = logging.getLogger(__name__)

# 中文/英文句末标点 —— 切分的绝对硬边界
_SENTENCE_TERMINATORS = re.compile(r"[。；;.!！?？\n]")


# ────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────
@dataclass
class TextChunk:
    """三粒度索引的最小存储单元"""
    content: str                    # 切片文本
    granularity: Literal[128, 512, 1024]  # 所属粒度层
    token_count: int                # 实际 token 数
    chunk_index: int                # 在同粒度内的序号
    metadata: BlockMetadata         # 继承自父 ParsedBlock 的溯源元数据
    overlap_prev: str = ""          # 与前一个切片的重叠文本 (用于上下文衔接)


# ────────────────────────────────────────────
# 语义切分器
# ────────────────────────────────────────────
class SemanticChunker:
    """
    基于医学标点硬边界的三粒度语义切分器

    使用方法:
        from app.knowledge.parser import DualTrackMedicalParser
        from app.knowledge.chunker import SemanticChunker

        parser = DualTrackMedicalParser()
        blocks = parser.parse("儿科指南.pdf")

        chunker = SemanticChunker()
        all_chunks = chunker.chunk_all_granularities(blocks)
        # all_chunks = {128: [...], 512: [...], 1024: [...]}
    """

    # 三粒度配置: (目标token数, 重叠token数)
    GRANULARITIES = {
        128:  (128,  32),   # 细节层: 约 2-3 句话, 重叠 1 句
        512:  (512,  64),   # 概念层: 约 1 段话, 重叠 2 句
        1024: (1024, 128),  # 上下文层: 约 2-3 段话, 重叠 1 段
    }

    def __init__(self, char_per_token: float = 1.5):
        """
        Args:
            char_per_token: 每个 token 对应的平均字符数
                           中文约 1字≈1.5-2 tokens, 这里取保守值
        """
        self._char_per_token = char_per_token

    # ── 公共入口 ──
    def chunk_all_granularities(
        self, blocks: list[ParsedBlock]
    ) -> dict[int, list[TextChunk]]:
        """
        对所有 ParsedBlock 执行三粒度切分

        Returns:
            {128: [TextChunk, ...], 512: [...], 1024: [...]}
        """
        result = {}
        for granularity, (target_tokens, overlap_tokens) in self.GRANULARITIES.items():
            chunks = self._chunk_blocks(blocks, target_tokens, overlap_tokens, granularity)
            result[granularity] = chunks
            logger.info(
                f"[Chunker] 粒度 {granularity}: 生成 {len(chunks)} 个切片"
            )
        return result

    # ── 核心切分逻辑 ──
    def _chunk_blocks(
        self,
        blocks: list[ParsedBlock],
        target_tokens: int,
        overlap_tokens: int,
        granularity: int,
    ) -> list[TextChunk]:
        """对全部块执行指定粒度的切分"""
        all_chunks: list[TextChunk] = []

        for block in blocks:
            # 表格块: 整体保留, 不做二次切分 (防止剂量表格错乱)
            if block.metadata.block_type == "table":
                all_chunks.append(TextChunk(
                    content=block.content,
                    granularity=granularity,
                    token_count=self._estimate_tokens(block.content),
                    chunk_index=len(all_chunks),
                    metadata=block.metadata,
                ))
                continue

            # 文本块: 执行语义切分
            sentences = self._split_into_sentences(block.content)
            if not sentences:
                continue

            sub_chunks = self._merge_sentences_to_chunks(
                sentences=sentences,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                metadata=block.metadata,
                granularity=granularity,
                start_index=len(all_chunks),
            )
            all_chunks.extend(sub_chunks)

        return all_chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        按医学标点将文本拆分为句子列表

        硬边界: 。；;.!！?？\\n
        保留标点符号在句尾 (不丢弃)
        """
        # 在标点后切分, 但保留标点
        parts = _SENTENCE_TERMINATORS.split(text)
        delimiters = _SENTENCE_TERMINATORS.findall(text)

        sentences = []
        for i, part in enumerate(parts):
            s = part.strip()
            # 把标点符号重新拼回句尾
            if i < len(delimiters):
                s += delimiters[i]
            s = s.strip()
            if s:
                sentences.append(s)

        return sentences

    def _merge_sentences_to_chunks(
        self,
        sentences: list[str],
        target_tokens: int,
        overlap_tokens: int,
        metadata: BlockMetadata,
        granularity: int,
        start_index: int,
    ) -> list[TextChunk]:
        """
        将句子列表合并为目标 token 长度的切片

        核心规则:
          - 逐句累加, 直到接近目标长度
          - 绝不在句子中间截断
          - 宁可短于目标, 也不截断半句话
          - 相邻切片之间保留重叠区 (overlap)
        """
        if not sentences:
            return []

        chunks: list[TextChunk] = []
        # 双指针: start 是当前切片起始句索引, end 往前探
        start = 0

        while start < len(sentences):
            # 从 start 开始, 贪心地往后累加句子
            current_tokens = 0
            end = start

            while end < len(sentences):
                sent_tokens = self._estimate_tokens(sentences[end])
                if current_tokens + sent_tokens > target_tokens and end > start:
                    # 超出目标长度且已有内容, 停止累加
                    break
                current_tokens += sent_tokens
                end += 1

            # [start, end) 构成一个切片
            chunk_text = "".join(sentences[start:end])

            # 计算前一切片的重叠文本
            overlap_text = ""
            if chunks:
                prev_start = max(0, start - 1)
                overlap_sents = []
                overlap_tok = 0
                for j in range(start - 1, -1, -1):
                    s_tok = self._estimate_tokens(sentences[j])
                    if overlap_tok + s_tok > overlap_tokens:
                        break
                    overlap_sents.insert(0, sentences[j])
                    overlap_tok += s_tok
                overlap_text = "".join(overlap_sents)

            chunks.append(TextChunk(
                content=chunk_text,
                granularity=granularity,
                token_count=current_tokens,
                chunk_index=start_index + len(chunks),
                metadata=metadata,
                overlap_prev=overlap_text,
            ))

            # 计算下一个切片的起始位置 (减去重叠句数)
            overlap_sent_count = 0
            overlap_tok = 0
            for j in range(end - 1, start - 1, -1):
                s_tok = self._estimate_tokens(sentences[j])
                if overlap_tok + s_tok > overlap_tokens:
                    break
                overlap_sent_count += 1
                overlap_tok += s_tok

            # 下一个切片从 (end - overlap_sent_count) 开始
            # 但必须至少前进 1 句, 防止死循环
            next_start = max(start + 1, end - overlap_sent_count)
            start = next_start

        return chunks

    def _extract_overlap(
        self, sentences: list[str], overlap_tokens: int
    ) -> list[str]:
        """从句子列表尾部提取不超过 overlap_tokens 的重叠句"""
        overlap: list[str] = []
        total = 0
        for sent in reversed(sentences):
            sent_tokens = self._estimate_tokens(sent)
            if total + sent_tokens > overlap_tokens:
                break
            overlap.insert(0, sent)
            total += sent_tokens
        return overlap

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数"""
        return max(1, int(len(text) / self._char_per_token))
