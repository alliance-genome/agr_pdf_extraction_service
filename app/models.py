import os
import threading
import logging

from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, Text, Index, ForeignKey, create_engine, JSON
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
    extract_images = Column(Boolean, nullable=False, default=False)
    review_images = Column(Boolean, nullable=False, default=False)
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


class PipelineJob(Base):
    __tablename__ = "pipeline_job"
    process_id = Column(UUID(as_uuid=False).with_variant(String(36), "sqlite"), ForeignKey("extraction_run.process_id"), primary_key=True)
    stage = Column(String, nullable=False, default="extraction")
    state = Column(String, nullable=False, default="ready")
    generation = Column(Integer, nullable=False, default=1)
    attempts = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False)
    deadline_at = Column(DateTime(timezone=True), nullable=False)
    dispatched_at = Column(DateTime(timezone=True))
    heartbeat_at = Column(DateTime(timezone=True))
    stage_started_at = Column(DateTime(timezone=True))
    extraction_ended_at = Column(DateTime(timezone=True))
    provider_started = Column(Boolean, nullable=False, default=False)
    source = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    checkpoint = Column(JSON().with_variant(JSONB, "postgresql"))
    options = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    progress = Column(JSON().with_variant(JSONB, "postgresql"))
    result = Column(JSON().with_variant(JSONB, "postgresql"))


Index("idx_pipeline_job_dispatch", PipelineJob.state, PipelineJob.stage, PipelineJob.created_at, postgresql_where=PipelineJob.state != "terminal")


TERMINAL_EXTRACTION_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def update_extraction_run_if_nonterminal(session, process_id, values):
    """Atomically update a run only while no terminal transition has won.

    Worker finalization, cancellation, and late nonterminal writes share this
    conditional update so the first committed terminal state is immutable.
    Returns ``(updated, current_run)`` after committing the attempted change.
    """
    updated = (
        session.query(ExtractionRun)
        .filter(
            ExtractionRun.process_id == process_id,
            ExtractionRun.status.notin_(TERMINAL_EXTRACTION_STATUSES),
        )
        .update(values, synchronize_session=False)
    )
    session.commit()
    session.expire_all()
    return updated == 1, session.get(ExtractionRun, process_id)


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

        if database_url.startswith("postgresql"):
            engine_kwargs["connect_args"] = {
                "connect_timeout": Config.PIPELINE_DB_TIMEOUT_SECONDS,
                "options": f"-c statement_timeout={Config.PIPELINE_DB_TIMEOUT_SECONDS * 1000}",
                "keepalives": 1, "keepalives_idle": Config.PIPELINE_DB_TIMEOUT_SECONDS,
                "keepalives_interval": 1, "keepalives_count": 3,
            }
        _ENGINE = create_engine(database_url, **engine_kwargs)
        _SESSION_FACTORY = scoped_session(sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False))
        _DATABASE_URL = database_url

        logger.info("Initialized SQLAlchemy database engine")
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
