"""Check the backend PostgreSQL driver contract without contacting a database.

Run during both image builds so a dependency/default-driver change cannot first
surface during AMI provisioning. Match both application and Alembic factories.
"""

from sqlalchemy import create_engine, engine_from_config


def check_database_driver():
    for drivername in ("postgresql", "postgresql+psycopg2"):
        url = f"{drivername}://localhost/pdfx_driver_check"
        for factory in (
            lambda: create_engine(url),
            lambda: engine_from_config({"sqlalchemy.url": url}, prefix="sqlalchemy."),
        ):
            engine = factory()
            try:
                if engine.dialect.dbapi.__name__ != "psycopg2":
                    raise RuntimeError(
                        "PDFX requires Psycopg 2 for application and migration engines"
                    )
            finally:
                engine.dispose()
    print("PostgreSQL application and migration driver checks passed (psycopg2).")


if __name__ == "__main__":
    check_database_driver()
