import json
from pathlib import Path

import pytest

from prepare_kb_sources import compute_sha256
from rebuild_index import _guideline_paths


def _write_manifest(guideline_dir: Path, sources: list[dict]) -> None:
    guideline_dir.mkdir(parents=True, exist_ok=True)
    (guideline_dir / "source_manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "sources": sources}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_guideline_paths_uses_only_manifest_approved_sources(tmp_path):
    guideline_dir = tmp_path / "data" / "guidelines"
    approved_pdf = guideline_dir / "approved.pdf"
    ignored_pdf = guideline_dir / "ignored.pdf"
    approved_pdf.parent.mkdir(parents=True)
    approved_pdf.write_bytes(b"%PDF approved")
    ignored_pdf.write_bytes(b"%PDF ignored")

    _write_manifest(
        guideline_dir,
        [
            {
                "source_id": "SRC-001",
                "filename": approved_pdf.name,
                "status": "approved",
                "included_in_kb": True,
                "sha256": compute_sha256(approved_pdf),
            },
            {
                "source_id": "SRC-002",
                "filename": ignored_pdf.name,
                "status": "inspected",
                "included_in_kb": False,
                "sha256": compute_sha256(ignored_pdf),
            },
        ],
    )

    assert _guideline_paths(tmp_path) == [approved_pdf]


def test_guideline_paths_rejects_included_source_without_approval(tmp_path):
    guideline_dir = tmp_path / "data" / "guidelines"
    pdf_path = guideline_dir / "pending.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF pending")

    _write_manifest(
        guideline_dir,
        [
            {
                "source_id": "SRC-003",
                "filename": pdf_path.name,
                "status": "inspected",
                "included_in_kb": True,
                "sha256": compute_sha256(pdf_path),
            },
        ],
    )

    with pytest.raises(ValueError, match="approved/indexed"):
        _guideline_paths(tmp_path)


def test_guideline_paths_rejects_hash_mismatch(tmp_path):
    guideline_dir = tmp_path / "data" / "guidelines"
    pdf_path = guideline_dir / "changed.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF original")

    _write_manifest(
        guideline_dir,
        [
            {
                "source_id": "SRC-004",
                "filename": pdf_path.name,
                "status": "approved",
                "included_in_kb": True,
                "sha256": "0" * 64,
            },
        ],
    )

    with pytest.raises(ValueError, match="SHA-256"):
        _guideline_paths(tmp_path)
