"""
审计当前 Chroma 索引质量，确认来源文件、页码分布与噪声过滤状态。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import chromadb


_REFERENCE_HEADING_RE = re.compile(r"^\s*##\s*[［\[]?\s*参\s*考\s*文\s*献", re.MULTILINE)
_PICTURE_PLACEHOLDER_RE = re.compile(
    r"\*\*==>\s*picture\s*\[[^\]]+\]\s*intentionally omitted\s*<==\*\*",
    re.IGNORECASE,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _collection_names() -> list[str]:
    return ["detail_128", "concept_512", "context_1024"]


def _expected_sources(root: Path) -> list[str]:
    guideline_dir = root / "data" / "guidelines"
    return sorted(pdf.name for pdf in guideline_dir.glob("*.pdf"))


def _load_index_status(chroma_dir: Path) -> dict[str, object]:
    status_path = chroma_dir / "index_status.json"
    if not status_path.exists():
        return {
            "ready": False,
            "reason": f"{status_path.name} missing",
        }

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"failed to read {status_path.name}: {exc}",
        }
    return status if isinstance(status, dict) else {"ready": False, "reason": "index_status.json is not a JSON object"}


def _audit_collection(collection) -> dict[str, object]:
    data = collection.get(include=["metadatas", "documents"])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", [])

    source_counter: Counter[str] = Counter()
    page_buckets: dict[str, list[int]] = defaultdict(list)
    reference_like = 0
    picture_noise = 0

    for meta, doc in zip(metadatas, documents):
        source = meta.get("source_file", "")
        page = int(meta.get("page_number", 0) or 0)
        source_counter[source] += 1
        if page:
            page_buckets[source].append(page)
        if _REFERENCE_HEADING_RE.search(doc or ""):
            reference_like += 1
        if _PICTURE_PLACEHOLDER_RE.search(doc or ""):
            picture_noise += 1

    page_ranges = {
        source: {
            "min": min(pages) if pages else None,
            "max": max(pages) if pages else None,
            "count": len(pages),
        }
        for source, pages in page_buckets.items()
    }

    return {
        "count": collection.count(),
        "sources": dict(source_counter),
        "page_ranges": page_ranges,
        "reference_like_documents": reference_like,
        "picture_noise_documents": picture_noise,
    }


def main() -> None:
    root = _project_root()
    chroma_dir = root / "backend" / "data" / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_dir))

    report = {}
    for name in _collection_names():
        try:
            collection = client.get_collection(name)
            report[name] = _audit_collection(collection)
        except Exception as exc:
            report[name] = {
                "count": 0,
                "sources": {},
                "page_ranges": {},
                "reference_like_documents": 0,
                "picture_noise_documents": 0,
                "error": str(exc),
            }

    expected_sources = set(_expected_sources(root))
    actual_sources = set(report.get("detail_128", {}).get("sources", {}).keys())
    report["completeness"] = {
        "expected_sources": sorted(expected_sources),
        "actual_sources": sorted(actual_sources),
        "missing_sources": sorted(expected_sources - actual_sources),
        "ready": expected_sources == actual_sources,
    }
    report["index_status"] = _load_index_status(chroma_dir)

    report_path = root / "docs" / "index_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n索引审计报告已写入: {report_path}")


if __name__ == "__main__":
    main()
