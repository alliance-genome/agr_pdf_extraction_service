"""create extraction_run table

Revision ID: b1c2d3e4f5a6
Revises:
Create Date: 2026-02-11 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "b1c2d3e4f5a6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "extraction_run",
        sa.Column("process_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("reference_curie", sa.String(), nullable=True),
        sa.Column("mod_abbreviation", sa.String(), nullable=True),
        sa.Column("source_pdf_md5", sa.String(), nullable=True),
        sa.Column("source_referencefile_id", sa.Integer(), nullable=True),
        sa.Column("config_version", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("artifacts_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("log_s3_key", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("process_id"),
    )
    op.create_index("idx_extraction_run_curie", "extraction_run", ["reference_curie"])
    op.create_index("idx_extraction_run_md5", "extraction_run", ["source_pdf_md5"])
    op.create_index("idx_extraction_run_status", "extraction_run", ["status"])


def downgrade():
    op.drop_index("idx_extraction_run_status", table_name="extraction_run")
    op.drop_index("idx_extraction_run_md5", table_name="extraction_run")
    op.drop_index("idx_extraction_run_curie", table_name="extraction_run")
    op.drop_table("extraction_run")
