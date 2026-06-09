import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from prepare_kb_sources import (
    download_registered_sources,
    inspect_staged_sources,
    load_manifest,
)


def _write_manifest(path: Path, sources: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": "1.0", "sources": sources}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_load_manifest_rejects_duplicate_source_ids(tmp_path):
    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"source_id": "SRC-005", "status": "registered"},
            {"source_id": "SRC-005", "status": "registered"},
        ],
    )

    with pytest.raises(ValueError, match="重复 source_id"):
        load_manifest(manifest_path)


def test_inspect_staged_sources_records_hash_and_parseability(tmp_path):
    staging_dir = tmp_path / "_staging"
    staging_dir.mkdir()
    pdf_path = staging_dir / "guideline.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test-content")

    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "source_id": "SRC-005",
                "title": "测试指南",
                "filename": pdf_path.name,
                "status": "downloaded",
            }
        ],
    )

    class FakeParser:
        def inspect_source(self, path):
            assert Path(path) == pdf_path
            return SimpleNamespace(
                page_count=12,
                sampled_pages=12,
                text_pages=11,
                image_pages=2,
                scan_heavy=False,
            )

    result = inspect_staged_sources(
        manifest_path=manifest_path,
        staging_dir=staging_dir,
        parser=FakeParser(),
    )

    source = result["sources"][0]
    assert source["status"] == "inspected"
    assert source["file_size"] == len(b"%PDF-1.4 test-content")
    assert len(source["sha256"]) == 64
    assert source["inspection"]["page_count"] == 12
    assert source["inspection"]["scan_heavy"] is False


def test_inspect_staged_sources_rejects_scan_heavy_pdf(tmp_path):
    staging_dir = tmp_path / "_staging"
    staging_dir.mkdir()
    pdf_path = staging_dir / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 scan")

    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "source_id": "SRC-006",
                "title": "扫描指南",
                "filename": pdf_path.name,
                "status": "downloaded",
            }
        ],
    )

    class FakeParser:
        def inspect_source(self, path):
            return SimpleNamespace(
                page_count=20,
                sampled_pages=20,
                text_pages=0,
                image_pages=20,
                scan_heavy=True,
            )

    result = inspect_staged_sources(
        manifest_path=manifest_path,
        staging_dir=staging_dir,
        parser=FakeParser(),
    )

    source = result["sources"][0]
    assert source["status"] == "rejected_scan_heavy"
    assert source["inspection"]["scan_heavy"] is True


def test_download_registered_sources_records_pdf_metadata(tmp_path):
    manifest_path = tmp_path / "source_manifest.json"
    staging_dir = tmp_path / "_staging"
    pdf_content = b"%PDF-1.7 official-guideline"
    _write_manifest(
        manifest_path,
        [
            {
                "source_id": "SRC-005",
                "title": "测试指南",
                "filename": "guideline.pdf",
                "download_url": "https://example.test/guideline.pdf",
                "status": "registered",
                "inspection": {"page_count": 999, "scan_heavy": True},
                "inspection_error": "旧体检结果",
            }
        ],
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=pdf_content,
            request=request,
        )
    )
    with httpx.Client(transport=transport) as client:
        result = download_registered_sources(
            manifest_path=manifest_path,
            staging_dir=staging_dir,
            client=client,
        )

    source = result["sources"][0]
    assert source["status"] == "downloaded"
    assert source["file_size"] == len(pdf_content)
    assert len(source["sha256"]) == 64
    assert source["downloaded_at"]
    assert "inspection" not in source
    assert "inspection_error" not in source
    assert (staging_dir / "guideline.pdf").read_bytes() == pdf_content


def test_download_registered_sources_rejects_non_pdf_response(tmp_path):
    manifest_path = tmp_path / "source_manifest.json"
    staging_dir = tmp_path / "_staging"
    _write_manifest(
        manifest_path,
        [
            {
                "source_id": "SRC-006",
                "title": "错误下载",
                "filename": "not-a-pdf.pdf",
                "download_url": "https://example.test/not-a-pdf.pdf",
                "status": "registered",
            }
        ],
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>not a pdf</html>",
            request=request,
        )
    )
    with httpx.Client(transport=transport) as client:
        result = download_registered_sources(
            manifest_path=manifest_path,
            staging_dir=staging_dir,
            client=client,
        )

    source = result["sources"][0]
    assert source["status"] == "download_failed"
    assert "PDF" in source["download_error"]
    assert not (staging_dir / "not-a-pdf.pdf").exists()
