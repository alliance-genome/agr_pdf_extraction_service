"""Exercise authenticated HTTP delivery with a cold backend and durable receipts."""
import io
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from app.artifacts import ArtifactReader, METHODS
from .test_main import _patch_singletons, client

PID = "00000000-0000-0000-0000-000000000123"
AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def storage(monkeypatch):
    import app.main as main
    run = {"status": "complete", "artifacts_json": {method: f"pdfx/audit/{PID}/{method}" for method in METHODS},
           "log_s3_key": f"pdfx/audit/{PID}/log.ndjson"}
    run["artifacts_json"]["images"] = [{"filename": "_page_0_Figure_1.jpeg", "s3_key": f"pdfx/audit/{PID}/image.jpeg", "size_bytes": 100, "caption_text": "a caption with / slash"}]
    monkeypatch.setattr(main.status_reader, "lookup", MagicMock(return_value=(True, run)))
    monkeypatch.setattr(main.settings, "ARTIFACT_S3_BUCKET", "artifacts")
    monkeypatch.setattr(main.settings, "ARTIFACT_STREAM_CHUNK_BYTES", 7)
    s3 = MagicMock()
    bodies = []
    def get(**kwargs):
        body = io.BytesIO(b"durable bytes " * 10)
        bodies.append(body)
        return {"Body": body, "ContentLength": 140, "ContentType": "text/markdown", "ContentDisposition": 'attachment; filename="output.md"'}
    s3.get_object.side_effect = get
    s3.generate_presigned_url.return_value = "https://storage.invalid/signed"
    monkeypatch.setattr(ArtifactReader, "client", lambda self: s3)
    yield run, s3, bodies
    main.lifecycle.ensure_running.assert_not_called()
    main.lifecycle.touch.assert_not_called()
    assert main.job_queue.size == 0
    assert all(b.closed for b in bodies)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_completed_outputs_stream_when_gpu_stopped(client, storage, method):
    response = client.get(f"/api/v1/extract/{PID}/download/{method}", headers=AUTH)
    assert response.status_code == 200
    assert response.content == b"durable bytes " * 10
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == 'attachment; filename="output.md"'
    storage[1].get_object.assert_called_once_with(Bucket="artifacts", Key=f"pdfx/audit/{PID}/{method}")


@pytest.mark.parametrize("suffix", ["images", "images/urls", "artifacts", "artifacts/urls", "logs"])
def test_manifests_without_gpu(client, storage, suffix):
    response = client.get(f"/api/v1/extract/{PID}/{suffix}", headers=AUTH)
    assert response.status_code == 200
    if suffix == "images":
        image = response.json()["images"][0]
        assert image["page_index"] == 0
        assert image["caption_text"] == "a caption with / slash"
    assert not any("caption" in str(call) for call in storage[1].generate_presigned_url.call_args_list)


def test_image_bytes(client, storage):
    response = client.get(f"/api/v1/extract/{PID}/images/_page_0_Figure_1.jpeg", headers=AUTH)
    assert response.status_code == 200
    assert response.content == b"durable bytes " * 10


@pytest.mark.parametrize("suffix", ["download/grobid", "images", "images/urls", "images/figure.png", "artifacts", "artifacts/urls", "logs"])
def test_auth_before_storage(client, storage, suffix):
    import app.main as main
    from app.auth import CognitoAuth
    main.cognito_auth = CognitoAuth()
    response = client.get(f"/api/v1/extract/{PID}/{suffix}")
    assert response.status_code == 401
    main.status_reader.lookup.assert_not_called()
    storage[1].get_object.assert_not_called()


@pytest.mark.parametrize("state,code", [("started",409),("pending",409),("failed",409),("cancelled",409)])
def test_unfinished_jobs(client, storage, state, code):
    storage[0]["status"] = state
    assert client.get(f"/api/v1/extract/{PID}/download/grobid", headers=AUTH).status_code == code
    storage[1].get_object.assert_not_called()


@pytest.mark.parametrize("available,run,code", [(False,None,503),(True,None,404)])
def test_unavailable_or_unknown_metadata(client, storage, monkeypatch, available, run, code):
    import app.main as main
    main.status_reader.lookup.return_value = available, run
    assert client.get(f"/api/v1/extract/{PID}/download/grobid", headers=AUTH).status_code == code


@pytest.mark.parametrize("suffix,code", [("download/nope",400),("images/missing.png",404)])
def test_invalid_or_missing(client, storage, suffix, code):
    assert client.get(f"/api/v1/extract/{PID}/{suffix}", headers=AUTH).status_code == code


def test_invalid_id(client, storage):
    assert client.get("/api/v1/extract/not-a-uuid/download/grobid", headers=AUTH).status_code == 400


@pytest.mark.parametrize("error,code", [("NoSuchKey",404),("AccessDenied",503),("SlowDown",503)])
def test_storage_failures(client, storage, error, code):
    storage[1].get_object.side_effect = ClientError({"Error":{"Code":error}}, "GetObject")
    assert client.get(f"/api/v1/extract/{PID}/download/grobid", headers=AUTH).status_code == code


@pytest.mark.parametrize("key", ["other/secret", "pdfx/audit/../secret", "https://host/private", {"key":"bad"}])
def test_invalid_receipt_never_reads_s3(client, storage, key):
    storage[0]["artifacts_json"]["grobid"] = key
    assert client.get(f"/api/v1/extract/{PID}/download/grobid", headers=AUTH).status_code == 503
    storage[1].get_object.assert_not_called()


def test_missing_merge_receipt_is_internal_error(client, storage):
    del storage[0]["artifacts_json"]["merged"]
    storage[0]["consensus_metrics_json"] = {}
    assert client.get(f"/api/v1/extract/{PID}/download/merged", headers=AUTH).status_code == 500


def test_repeat_without_local_state(client, storage):
    import app.main as main
    for _ in range(2):
        main.proxy_to_backend_process.clear()
        main.job_trackers.clear()
        assert client.get(f"/api/v1/extract/{PID}/download/grobid", headers=AUTH).status_code == 200


def test_mapped_id_preserves_public_manifest_id(client, storage):
    import app.main as main
    alias = "00000000-0000-0000-0000-000000000999"
    main.proxy_to_backend_process[alias] = PID
    response = client.get(f"/api/v1/extract/{alias}/images", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["process_id"] == alias
    main.status_reader.lookup.assert_called_once_with(PID)


def test_source_pdf_url_is_retained(client, storage):
    storage[0]["artifacts_json"]["source_pdf"] = f"pdfx/audit/{PID}/source.pdf"
    response = client.get(f"/api/v1/extract/{PID}/artifacts/urls", headers=AUTH)
    assert any(entry["name"] == "$.source_pdf" for entry in response.json()["artifact_urls"])


@pytest.mark.asyncio
async def test_disconnect_closes_body_before_iteration():
    from app.artifacts import ArtifactStreamResponse
    body = io.BytesIO(b"data")
    async def receive():
        return {"type": "http.disconnect"}
    async def send(message):
        raise OSError("connection closed")
    response = ArtifactStreamResponse(body, iter([b"data"]))
    with pytest.raises(Exception):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert body.closed


def test_actual_s3_client_is_bounded(monkeypatch):
    import app.artifacts as module
    monkeypatch.setattr(module.settings, "ARTIFACT_S3_BUCKET", "artifacts")
    factory = MagicMock()
    monkeypatch.setattr(module, "_s3_client", None)
    session = MagicMock()
    session.return_value.client = factory
    monkeypatch.setattr(module.boto3, "Session", session)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        clients = list(pool.map(lambda _: ArtifactReader(None).client(), range(20)))
    assert all(client is clients[0] for client in clients)
    session.assert_called_once()
    factory.assert_called_once()
    config = factory.call_args.kwargs["config"]
    assert config.connect_timeout == module.settings.ARTIFACT_S3_TIMEOUT_SECONDS
    assert config.read_timeout == module.settings.ARTIFACT_S3_TIMEOUT_SECONDS
    assert config.retries["total_max_attempts"] == module.settings.ARTIFACT_S3_MAX_ATTEMPTS


def test_native_receipt_manifest_preserves_nested_paths_without_media_types(client, storage):
    fields = ("artifact", "manifest", "page_provenance", "style")
    storage[0]["artifacts_json"]["native"] = {method: {
        **{field: f"pdfx/audit/{PID}/native/{method}/{field}" for field in fields},
        "media_type": "application/json", "sha256": "deadbeef",
    } for method in ("grobid", "docling", "marker")}
    response = client.get(f"/api/v1/extract/{PID}/artifacts/urls", headers=AUTH)
    assert response.status_code == 200
    entries = response.json()["artifact_urls"]
    expected = {f"$.native.{method}.{field}" for method in ("grobid", "docling", "marker") for field in fields}
    assert expected <= {entry["name"] for entry in entries}
    assert not any("media_type" in entry["name"] or "sha256" in entry["name"] for entry in entries)


def test_s3_client_initialization_failure_is_503(client, storage, monkeypatch):
    monkeypatch.setattr(ArtifactReader, "client", MagicMock(side_effect=RuntimeError("credential provider unavailable")))
    assert client.get(f"/api/v1/extract/{PID}/download/grobid", headers=AUTH).status_code == 503
