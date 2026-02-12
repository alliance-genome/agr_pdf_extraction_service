import uuid

import pytest

from app.models import Base, ExtractionRun, get_engine, get_session, reset_db_engine


@pytest.fixture(autouse=True)
def sqlite_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_db_engine()
    Base.metadata.create_all(bind=get_engine())
    yield
    reset_db_engine()


def test_extraction_run_insert_and_query():
    session = get_session()

    process_id = str(uuid.uuid4())
    session.add(ExtractionRun(
        process_id=process_id,
        reference_curie="PMID:12345",
        mod_abbreviation="ZFIN",
        source_pdf_md5="abc123",
        status="queued",
    ))
    session.commit()

    run = session.get(ExtractionRun, process_id)
    assert run is not None
    assert run.process_id == process_id
    assert run.reference_curie == "PMID:12345"
    assert run.mod_abbreviation == "ZFIN"
    assert run.source_pdf_md5 == "abc123"
    assert run.status == "queued"

    session.close()


def test_extraction_run_default_status():
    session = get_session()

    process_id = str(uuid.uuid4())
    session.add(ExtractionRun(process_id=process_id))
    session.commit()

    run = session.get(ExtractionRun, process_id)
    assert run is not None
    assert run.status == "queued"

    session.close()
