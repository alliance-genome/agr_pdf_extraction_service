"""Fail deployment before runtime startup if the app role lacks stage access."""

from sqlalchemy import text
from app.models import get_engine


def check_pipeline_database(connection):
    if connection.dialect.name != "postgresql":
        raise RuntimeError("The staged runtime requires PostgreSQL")
    missing = []
    for table in ("extraction_run", "pipeline_job"):
        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            allowed = connection.execute(
                text("SELECT has_table_privilege(current_user, :table, :privilege)"),
                {"table": table, "privilege": privilege},
            ).scalar()
            if not allowed:
                missing.append(f"{table}:{privilege}")
    if missing:
        raise RuntimeError(
            "Runtime database role needs explicit table grants after migration: "
            + ", ".join(missing)
        )


if __name__ == "__main__":
    with get_engine().connect() as connection:
        check_pipeline_database(connection)
    print("Runtime database privileges verified for extraction and pipeline tables.")
