"""知识库资料准入工具：登记、体检并记录可追溯元数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.knowledge.parser import DualTrackMedicalParser


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """读取并校验资料清单，防止 source_id 冲突破坏追溯关系。"""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source_manifest.json 必须包含 sources 列表")

    seen_ids: set[str] = set()
    for source in sources:
        source_id = source.get("source_id")
        if not source_id:
            raise ValueError("每个资料条目都必须包含 source_id")
        if source_id in seen_ids:
            raise ValueError(f"重复 source_id: {source_id}")
        seen_ids.add(source_id)

    return manifest


def save_manifest(manifest_path: str | Path, manifest: dict[str, Any]) -> None:
    """以稳定、可审阅的 JSON 格式保存资料清单。"""
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def compute_sha256(file_path: str | Path) -> str:
    """流式计算文件哈希，避免大 PDF 一次性载入内存。"""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspection_to_dict(inspection: Any) -> dict[str, Any]:
    if is_dataclass(inspection):
        return asdict(inspection)
    return vars(inspection).copy()


def download_registered_sources(
    manifest_path: str | Path,
    staging_dir: str | Path,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """下载已登记且具有直链的资料，不接受非 PDF 响应。"""
    manifest = load_manifest(manifest_path)
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60.0),
        headers={"User-Agent": "Medaudit-RAG-KB-Admission/1.0"},
    )

    try:
        for source in manifest["sources"]:
            if source.get("status") != "registered":
                continue

            download_url = source.get("download_url")
            filename = source.get("filename")
            if not download_url or not filename:
                continue

            target_path = staging_path / filename
            try:
                response = http_client.get(download_url, follow_redirects=True)
                response.raise_for_status()
                content = response.content
                if not content.startswith(b"%PDF"):
                    content_type = response.headers.get("content-type", "unknown")
                    raise ValueError(
                        f"下载结果不是有效 PDF，content-type={content_type}"
                    )

                target_path.write_bytes(content)
                source["status"] = "downloaded"
                source["downloaded_at"] = datetime.now(timezone.utc).isoformat()
                source["resolved_download_url"] = str(response.url)
                source["file_size"] = target_path.stat().st_size
                source["sha256"] = compute_sha256(target_path)
                source.pop("download_error", None)
                source.pop("inspection", None)
                source.pop("inspection_error", None)
            except Exception as exc:
                if target_path.exists():
                    target_path.unlink()
                source["status"] = "download_failed"
                source["download_error"] = str(exc)
    finally:
        if own_client:
            http_client.close()

    save_manifest(manifest_path, manifest)
    return manifest


def inspect_staged_sources(
    manifest_path: str | Path,
    staging_dir: str | Path,
    parser: Any | None = None,
) -> dict[str, Any]:
    """体检 staging 中已下载的 PDF，并将结果写回清单。"""
    manifest = load_manifest(manifest_path)
    staging_path = Path(staging_dir)
    source_parser = parser or DualTrackMedicalParser()

    for source in manifest["sources"]:
        if source.get("status") != "downloaded":
            continue

        filename = source.get("filename")
        if not filename:
            source["status"] = "inspection_failed"
            source["inspection_error"] = "缺少 filename"
            continue

        pdf_path = staging_path / filename
        if not pdf_path.is_file():
            source["status"] = "inspection_failed"
            source["inspection_error"] = f"文件不存在: {pdf_path}"
            continue

        try:
            inspection = source_parser.inspect_source(pdf_path)
            inspection_data = _inspection_to_dict(inspection)
            source["file_size"] = pdf_path.stat().st_size
            source["sha256"] = compute_sha256(pdf_path)
            source["inspection"] = inspection_data
            source.pop("inspection_error", None)
            source["status"] = (
                "rejected_scan_heavy"
                if inspection_data.get("scan_heavy")
                else "inspected"
            )
        except Exception as exc:
            source["status"] = "inspection_failed"
            source["inspection_error"] = str(exc)

    save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库资料准入工具")
    parser.add_argument("command", choices=["download", "inspect"])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/guidelines/source_manifest.json"),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/guidelines/_staging"),
    )
    args = parser.parse_args()

    if args.command == "download":
        manifest = download_registered_sources(args.manifest, args.staging_dir)
        downloaded = sum(
            source.get("status") == "downloaded" for source in manifest["sources"]
        )
        failed = sum(
            source.get("status") == "download_failed"
            for source in manifest["sources"]
        )
        print(f"下载完成: 成功 {downloaded}，失败 {failed}")
    elif args.command == "inspect":
        manifest = inspect_staged_sources(args.manifest, args.staging_dir)
        inspected = sum(
            source.get("status") == "inspected" for source in manifest["sources"]
        )
        rejected = sum(
            source.get("status") == "rejected_scan_heavy"
            for source in manifest["sources"]
        )
        failed = sum(
            source.get("status") == "inspection_failed"
            for source in manifest["sources"]
        )
        print(f"体检完成: 可解析 {inspected}，扫描件拒绝 {rejected}，失败 {failed}")


if __name__ == "__main__":
    main()
