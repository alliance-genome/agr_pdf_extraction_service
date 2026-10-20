"""Exercise the database driver check against installed backend dependencies."""

from pathlib import Path
import runpy

import psycopg2

from app.models import get_engine, reset_db_engine


ROOT = Path(__file__).resolve().parents[1]


def test_image_driver_check_loads_both_factories_without_connecting(monkeypatch):
    def unexpected_connect(*args, **kwargs):
        raise AssertionError("the image build must not contact a database")

    monkeypatch.setattr(psycopg2, "connect", unexpected_connect)
    runpy.run_path(str(ROOT / "deploy/check_database_driver.py"), run_name="__main__")


def test_runtime_plain_postgres_url_uses_declared_driver(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/pdfx_driver_check")
    reset_db_engine()
    try:
        engine = get_engine()
        assert engine.dialect.dbapi is psycopg2
    finally:
        reset_db_engine()
