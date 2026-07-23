import pytest

from app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app()
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(tmp_path / "uploads")
    app.config["CACHE_FOLDER"] = str(tmp_path / "cache")
    with app.test_client() as test_client:
        yield test_client


def test_index_route_uses_canonical_async_api(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"/api/v1/extract" in response.data
    assert b"/process" not in response.data


def test_legacy_synchronous_process_route_is_removed(client):
    response = client.post("/process")

    assert response.status_code == 404


def test_openapi_route(client):
    response = client.get("/openapi.yaml")

    assert response.status_code == 200
    assert response.mimetype == "application/yaml"
