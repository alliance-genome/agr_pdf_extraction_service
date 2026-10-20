"""Bake qualification must exercise OCR and reject partial extractor success."""

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pypdfium2 as pdfium
import pytest

from deploy.aws.ami.warmup import positive_seconds, run_warmups, validate_result


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "deploy/aws/ami"
METHODS = ("grobid", "docling", "marker")


def result_for(methods=METHODS):
    return {
        "status": "success",
        "extraction_status": "complete_success",
        "available_extractors": list(methods),
        "failed_extractors": [],
        "extractor_failures": {},
        "cached_methods": [],
        "extractions": {m: "This paragraph exercises OCR." for m in methods},
    }


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {"extraction_status": "partial_success"},
        {"available_extractors": ["grobid", "marker"]},
        {"failed_extractors": ["docling"]},
        {"extractor_failures": {"docling": {"message": "model failed"}}},
        {"cached_methods": ["docling"]},
        {"extractions": {"grobid": "ok", "docling": " ", "marker": "ok"}},
    ],
)
def test_rejects_incomplete_or_cached_warmup(change):
    result = result_for()
    result.update(change)
    with pytest.raises(ValueError):
        validate_result(result, METHODS)


def test_scan_requires_recognized_text_not_merely_nonempty_output():
    result = result_for(("docling",))
    validate_result(result, ("docling",), require_ocr=True)
    result["extractions"]["docling"] = "<!-- image -->"
    with pytest.raises(ValueError, match="expected OCR text"):
        validate_result(result, ("docling",), require_ocr=True)


def test_scan_fixture_has_image_objects_and_no_selectable_text():
    with pdfium.PdfDocument(FIXTURES / "test-scan.pdf") as document:
        assert len(document) == 1
        page = document[0]
        text = page.get_textpage()
        try:
            assert not text.get_text_range().strip()
            assert any(
                obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE for obj in page.get_objects()
            )
        finally:
            text.close()
            page.close()


def test_submits_both_fixtures_without_paid_models_and_checks_worker_results():
    requests_seen = []

    def post(url, *, files, data, timeout):
        name, pdf, mime = files["file"]
        assert pdf.read().startswith(b"%PDF")
        requests_seen.append((name, copy.deepcopy(data)))
        assert url == "http://127.0.0.1:5000/api/v1/extract"
        assert mime == "application/pdf"
        assert timeout == 11
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"process_id": name},
        )

    celery = Mock()
    celery.AsyncResult.side_effect = [
        Mock(get=Mock(return_value=result_for())),
        Mock(get=Mock(return_value=result_for(("docling",)))),
    ]
    run_warmups(
        SimpleNamespace(post=post), celery, FIXTURES,
        job_timeout=123, http_timeout=11,
    )
    assert [item[0] for item in requests_seen] == ["test-sample.pdf", "test-scan.pdf"]
    assert [item[1]["methods"] for item in requests_seen] == [
        "grobid,docling,marker", "docling",
    ]
    for _, data in requests_seen:
        assert data["merge"] == data["extract_images"] == data["review_images"] == "false"
        assert data["clear_cache_scope"] == "extraction"
    assert [call.args[0] for call in celery.AsyncResult.call_args_list] == [
        "test-sample.pdf", "test-scan.pdf",
    ]


def test_worker_timeout_stops_bake_before_next_submission():
    response = Mock()
    response.json.return_value = {"process_id": "fixture-job"}
    session = Mock()
    session.post.return_value = response
    celery = Mock()
    celery.AsyncResult.return_value.get.side_effect = TimeoutError("worker timeout")
    with pytest.raises(TimeoutError):
        run_warmups(session, celery, FIXTURES, job_timeout=123, http_timeout=11)
    celery.AsyncResult.return_value.get.assert_called_once_with(timeout=123)
    assert session.post.call_count == 1


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_rejects_invalid_bake_timeout(monkeypatch, value):
    monkeypatch.setenv("PDFX_BAKE_WARMUP_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        positive_seconds("PDFX_BAKE_WARMUP_TIMEOUT_SECONDS", 600)


def test_cache_gate_still_requires_ocr_files(tmp_path):
    for directory in ("models", "rapidocr_models", "cache"):
        (tmp_path / "data" / directory).mkdir(parents=True)
    (tmp_path / "data/models/model.bin").write_bytes(b"model")
    (tmp_path / "data/cache/marker_worker_ready.json").write_text("{}")
    command = [
        "bash", "-c",
        'source "$1"; assert_caches_nonempty "$2"',
        "bake-check", str(FIXTURES / "provision.sh"), str(tmp_path),
    ]
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "rapidocr_models" in missing.stderr
    (tmp_path / "data/rapidocr_models/model.onnx").write_bytes(b"model")
    assert subprocess.run(command, capture_output=True).returncode == 0


def test_bake_copies_reviewed_helper_and_forwards_packer_limits():
    provision = (FIXTURES / "provision.sh").read_text()
    packer = (FIXTURES / "pdfx-backend.pkr.hcl").read_text()
    assert 'for fixture in warmup.py test-sample.pdf test-scan.pdf' in provision
    assert 'docker cp "$SERVICE_DIR/deploy/aws/ami/$fixture"' in provision
    assert 'pdfx-app python3.11 "$warmup_dir/warmup.py"' in provision
    for name in ("warmup", "http"):
        assert f"PDFX_BAKE_{name.upper()}_TIMEOUT_SECONDS='${{var.bake_{name}_timeout_seconds}}'" in packer


def test_fixture_and_helper_changes_trigger_backend_bake():
    workflow = (ROOT / ".github/workflows/main-build-and-deploy.yml").read_text()
    line = next(line for line in workflow.splitlines() if "if grep -Eq" in line and "warmup" in line)
    pattern = line.split("'", 2)[1]
    for filename in ("warmup.py", "test-sample.pdf", "test-scan.pdf"):
        check = subprocess.run(["grep", "-Eq", pattern], input=f"deploy/aws/ami/{filename}\n", text=True)
        assert check.returncode == 0
