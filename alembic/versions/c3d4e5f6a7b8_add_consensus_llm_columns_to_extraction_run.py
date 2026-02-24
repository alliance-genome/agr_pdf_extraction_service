"""add consensus/llm columns to extraction_run

Revision ID: c3d4e5f6a7b8
Revises: b1c2d3e4f5a6
Create Date: 2026-02-23 22:25:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c3d4e5f6a7b8"
down_revision = "b1c2d3e4f5a6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "extraction_run",
        sa.Column("consensus_metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "extraction_run",
        sa.Column("llm_usage_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "extraction_run",
        sa.Column("llm_cost_usd", sa.Numeric(precision=10, scale=6), nullable=True),
    )


def downgrade():
    op.drop_column("extraction_run", "llm_cost_usd")
    op.drop_column("extraction_run", "llm_usage_json")
    op.drop_column("extraction_run", "consensus_metrics_json")
