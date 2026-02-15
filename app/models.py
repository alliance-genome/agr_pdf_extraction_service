import os
import threading
import logging

from sqlalchemy import Column, DateTime, Integer, Numeric, String, Text, Index, create_engine, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

from config import Config

logger = logging.getLogger(__name__)

Base = declarative_base()


class ExtractionRun(Base):
    __tablename__ = "extraction_run"

    process_id = Column(UUID(as_uuid=False).with_variant(String(36), "sqlite"), primary_key=True)
    reference_curie = Column(String)
    mod_abbreviation = Column(String)
    source_pdf_md5 = Column(String)
    source_referencefile_id = Column(Integer)
    config_version = Column(String)
    status = Column(String, nullable=False, default="queued")
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    error_code = Column(String)
    error_message = Column(Text)
    artifacts_json = Column(JSON().with_variant(JSONB, "postgresql"))
    log_s3_key = Column(String)
    consensus_metrics_json = Column(JSON().with_variant(JSONB, "postgresql"))
    llm_usage_json = Column(JSON().with_variant(JSONB, "postgresql"))
    llm_cost_usd = Column(Numeric(precision=10, scale=6))


Index("idx_extraction_run_curie", ExtractionRun.reference_curie)
Index("idx_extraction_run_md5", ExtractionRun.source_pdf_md5)
Index("idx_extraction_run_status", ExtractionRun.status)


_ENGINE = None
_SESSION_FACTORY = None
_DATABASE_URL = None
_ENGINE_LOCK = threading.Lock()


def _resolve_database_url():
    return os.environ.get("DATABASE_URL", Config.DATABASE_URL)


def get_engine():
    global _ENGINE, _SESSION_FACTORY, _DATABASE_URL

    database_url = _resolve_database_url()

    with _ENGINE_LOCK:
        if _ENGINE is not None and _DATABASE_URL == database_url:
            return _ENGINE

        engine_kwargs = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in database_url:
                engine_kwargs["poolclass"] = StaticPool

        _ENGINE = create_engine(database_url, **engine_kwargs)
        _SESSION_FACTORY = scoped_session(sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False))
        _DATABASE_URL = database_url

        logger.info("Initialized SQLAlchemy engine for DATABASE_URL=%s", database_url)
        return _ENGINE


def get_session():
    global _SESSION_FACTORY

    if _SESSION_FACTORY is None:
        get_engine()
    return _SESSION_FACTORY()


def reset_db_engine():
    global _ENGINE, _SESSION_FACTORY, _DATABASE_URL

    with _ENGINE_LOCK:
        if _SESSION_FACTORY is not None:
            _SESSION_FACTORY.remove()
        if _ENGINE is not None:
            _ENGINE.dispose()

        _ENGINE = None
        _SESSION_FACTORY = None
        _DATABASE_URL = None


def create_all():
    Base.metadata.create_all(bind=get_engine())
