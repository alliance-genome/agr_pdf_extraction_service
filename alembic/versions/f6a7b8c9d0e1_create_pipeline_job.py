"""Add durable stage dispatch records. Drain old workers before runtime cutover."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "pipeline_job",
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("extraction_run.process_id"),
            primary_key=True,
        ),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        *[
            sa.Column(name, sa.DateTime(timezone=True))
            for name in (
                "dispatched_at",
                "heartbeat_at",
                "stage_started_at",
                "extraction_ended_at",
            )
        ],
        sa.Column("provider_started", sa.Boolean(), nullable=False),
        *[
            sa.Column(
                name,
                sa.JSON().with_variant(postgresql.JSONB, "postgresql"),
                nullable=name not in ("source", "options"),
            )
            for name in ("source", "checkpoint", "options", "progress", "result")
        ],
    )
    op.create_index(
        "idx_pipeline_job_dispatch",
        "pipeline_job",
        ["state", "stage", "created_at"],
        postgresql_where=sa.text("state != 'terminal'"),
    )


def downgrade():
    op.drop_index("idx_pipeline_job_dispatch", table_name="pipeline_job")
    op.drop_table("pipeline_job")
