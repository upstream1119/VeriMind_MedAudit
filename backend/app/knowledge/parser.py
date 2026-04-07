"""
VeriMind-Med 双轨 PDF 解析管线 (Dual-Track Medical PDF Parser)
================================================================
轨道 A: pymupdf4llm → 连续文本段落提取 (双栏自动校正)
轨道 B: pdfplumber  → 复杂医学表格识别与 Markdown 转录

设计原则:
  - 每一个输出块 (ParsedBlock) 都必须携带完整的溯源元数据
  - 表格数据绝不允许以"错乱长文本"的形式流入下游切分器
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber
import pymupdf4llm

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────
# 数据结构定义
# ────────────────────────────────────────────
@dataclass
class BlockMetadata:
    """每个解析块的强制溯源元数据"""
    source_file: str           # PDF 文件名
    source_hash: str           # 文件 SHA-256 (版本校验)
    page_number: int           # 所在页码
    chapter_title: str = ""    # 章节标题 (尽最大努力提取)
    block_type: Literal["text", "table"] = "text"


@dataclass
class ParsedBlock:
    """双轨解析的统一输出单元"""
    content: str               # Markdown 格式的文本内容
    metadata: BlockMetadata    # 溯源元数据
    token_estimate: int = 0    # 粗略 token 估算 (中文≈字数×2)


# ────────────────────────────────────────────
# 双轨解析器
# ────────────────────────────────────────────
class DualTrackMedicalParser:
    """
    双轨 PDF 解析器

    使用方法:
        parser = DualTrackMedicalParser()
        blocks = parser.parse("path/to/儿科指南.pdf")
        for block in blocks:
            print(block.content[:100], block.metadata)
    """

    def __init__(self, min_text_length: int = 20):
        """
        Args:
            min_text_length: 低于此字数的文本块将被丢弃 (过滤噪声)
        """
        self._min_text_length = min_text_length

    # ── 公共入口 ──
    def parse(self, pdf_path: str | Path) -> list[ParsedBlock]:
        """
        解析 PDF 文件, 返回带完整元数据的文档块列表

        双轨合并策略:
          1. 先走轨道 A (pymupdf4llm) 拿到全部连续文本
          2. 逐页走轨道 B (pdfplumber) 检测表格
          3. 将表格块插回到对应页码的文本流中
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        file_hash = self._compute_file_hash(pdf_path)
        logger.info(f"[Parser] 开始解析: {pdf_path.name} (SHA-256: {file_hash[:16]}...)")

        # 轨道 A: 连续文本
        text_blocks = self._track_a_text(pdf_path, file_hash)
        logger.info(f"[Parser] 轨道A完成: 提取了 {len(text_blocks)} 个文本块")

        # 轨道 B: 表格
        table_blocks = self._track_b_tables(pdf_path, file_hash)
        logger.info(f"[Parser] 轨道B完成: 提取了 {len(table_blocks)} 个表格块")

        # 合并: 按页码排序
        all_blocks = text_blocks + table_blocks
        all_blocks.sort(key=lambda b: b.metadata.page_number)

        logger.info(f"[Parser] 解析完成: 共 {len(all_blocks)} 个块")
        return all_blocks

    # ── 轨道 A: pymupdf4llm 文本提取 ──
    def _track_a_text(self, pdf_path: Path, file_hash: str) -> list[ParsedBlock]:
        """使用 pymupdf4llm 提取连续文本, 自动校正双栏排版"""
        blocks: list[ParsedBlock] = []

        # pymupdf4llm 返回按页组织的 Markdown 文本列表
        md_pages = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,  # 按页返回
        )

        for page_data in md_pages:
            page_num = page_data.get("metadata", {}).get("page", 0)
            text = page_data.get("text", "").strip()

            if len(text) < self._min_text_length:
                continue

            blocks.append(ParsedBlock(
                content=text,
                metadata=BlockMetadata(
                    source_file=pdf_path.name,
                    source_hash=file_hash,
                    page_number=page_num,
                    block_type="text",
                ),
                token_estimate=self._estimate_tokens(text),
            ))

        return blocks

    # ── 轨道 B: pdfplumber 表格提取 ──
    def _track_b_tables(self, pdf_path: Path, file_hash: str) -> list[ParsedBlock]:
        """使用 pdfplumber 检测并转录复杂医学表格为 Markdown 格式"""
        blocks: list[ParsedBlock] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue

                for table_idx, table in enumerate(tables):
                    md_table = self._table_to_markdown(table)
                    if not md_table:
                        continue

                    blocks.append(ParsedBlock(
                        content=md_table,
                        metadata=BlockMetadata(
                            source_file=pdf_path.name,
                            source_hash=file_hash,
                            page_number=page_idx + 1,
                            block_type="table",
                        ),
                        token_estimate=self._estimate_tokens(md_table),
                    ))

        return blocks

    # ── 工具函数 ──
    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        """
        将 pdfplumber 提取的二维表格转换为标准 Markdown 表格

        示例输出:
            | 年龄 | 体重(kg) | 剂量(mg/kg) |
            | --- | --- | --- |
            | <1岁 | 3-10 | 10-15 |
        """
        if not table or len(table) < 2:
            return ""

        # 清洗: None → 空字符串, 换行 → 空格
        cleaned = []
        for row in table:
            cleaned_row = [
                (cell or "").replace("\n", " ").strip()
                for cell in row
            ]
            cleaned.append(cleaned_row)

        # 构建 Markdown
        header = cleaned[0]
        col_count = len(header)

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * col_count) + " |",
        ]

        for row in cleaned[1:]:
            # 补齐列数不一致的情况
            padded = row + [""] * (col_count - len(row))
            lines.append("| " + " | ".join(padded[:col_count]) + " |")

        return "\n".join(lines)

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """计算文件 SHA-256 哈希 (用于版本校验)"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数 (中文约 1字≈2tokens, 英文约 1词≈1.3tokens)"""
        # 简单启发式: 总字符数 * 1.5 作为保守估计
        return int(len(text) * 1.5)
