"""知识库资料准入工具：登记、体检并记录可追溯元数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def approve_inspected_sources(
    manifest_path: str | Path,
    staging_dir: str | Path,
    formal_dir: str | Path,
    source_ids: list[str],
) -> dict[str, Any]:
    """将已完成体检和人工抽查的 staging PDF 批准进入正式知识库目录。"""
    manifest = load_manifest(manifest_path)
    staging_path = Path(staging_dir)
    formal_path = Path(formal_dir)
    formal_path.mkdir(parents=True, exist_ok=True)
    requested_ids = set(source_ids)
    approved_at = datetime.now(timezone.utc).isoformat()
    matched_ids: set[str] = set()

    for source in manifest["sources"]:
        source_id = source.get("source_id")
        if source_id not in requested_ids:
            continue
        matched_ids.add(source_id)

        if source.get("status") not in {"inspected", "approved", "indexed"}:
            raise ValueError(f"{source_id} 尚未完成可解析性体检，不能批准入库")
        if source.get("inspection", {}).get("scan_heavy"):
            raise ValueError(f"{source_id} 被判定为扫描件偏重，不能批准入库")
        if source.get("content_check", {}).get("status") != "spot_checked":
            raise ValueError(f"{source_id} 尚未完成 page/content 抽查，不能批准入库")

        filename = source.get("filename")
        expected_sha = source.get("sha256")
        if not filename or not expected_sha:
            raise ValueError(f"{source_id} 缺少 filename 或 sha256，不能批准入库")

        staged_pdf = staging_path / filename
        formal_pdf = formal_path / filename
        source_pdf = staged_pdf if staged_pdf.is_file() else formal_pdf
        if not source_pdf.is_file():
            raise FileNotFoundError(f"{source_id} 对应 PDF 不存在: {staged_pdf}")

        actual_sha = compute_sha256(source_pdf)
        if actual_sha != expected_sha:
            raise ValueError(f"{source_id} SHA-256 不匹配，不能批准入库")

        if source_pdf != formal_pdf:
            shutil.copy2(source_pdf, formal_pdf)

        source["status"] = "approved"
        source["included_in_kb"] = True
        source["approved_at"] = approved_at
        source.pop("approval_error", None)

    missing_ids = requested_ids - matched_ids
    if missing_ids:
        raise ValueError(f"manifest 中不存在 source_id: {sorted(missing_ids)}")

    save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库资料准入工具")
    parser.add_argument("command", choices=["download", "inspect", "approve"])
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
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=Path("data/guidelines"),
    )
    parser.add_argument("--source-id", action="append", default=[])
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
    elif args.command == "approve":
        if not args.source_id:
            raise SystemExit("approve 命令必须提供至少一个 --source-id")
        manifest = approve_inspected_sources(
            args.manifest,
            args.staging_dir,
            args.formal_dir,
            args.source_id,
        )
        approved = [
            source["source_id"]
            for source in manifest["sources"]
            if source.get("included_in_kb")
        ]
        print(f"批准入库完成: 当前正式知识库来源 {len(approved)} 份")


if __name__ == "__main__":
    main()
