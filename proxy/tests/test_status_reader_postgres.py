"""Opt-in joined-query validation against an isolated schema in the test DB."""

import os
import uuid

import psycopg2
from psycopg2 import sql
import pytest

from app.status_reader import RDSStatusReader


def test_joined_durable_status_and_read_only_active_work(monkeypatch):
    url = os.getenv("PDFX_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set PDFX_TEST_DATABASE_URL to an isolated PostgreSQL database")
    schema = "proxy_test_" + uuid.uuid4().hex
    pid = str(uuid.uuid4())
    admin = psycopg2.connect(url)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            cursor.execute(
                sql.SQL("SET search_path TO {}").format(sql.Identifier(schema))
            )
            for table in ("extraction_run", "pipeline_job"):
                cursor.execute(
                    sql.SQL("CREATE TABLE {} (LIKE public.{} INCLUDING ALL)").format(
                        sql.Identifier(table), sql.Identifier(table)
                    )
                )
            cursor.execute(
                "INSERT INTO extraction_run (process_id,status,extract_images,review_images) VALUES (%s,'queued',false,false)",
                (pid,),
            )
            cursor.execute(
                """INSERT INTO pipeline_job
                (process_id,stage,state,generation,attempts,created_at,deadline_at,provider_started,source,options)
                VALUES (%s,'finalization','ready',2,0,now(),now()+interval '1 hour',false,'{}','{}')""",
                (pid,),
            )
        reader = RDSStatusReader(url)
        monkeypatch.setattr(
            reader,
            "_connect",
            lambda: psycopg2.connect(
                url,
                options=f"-c search_path={schema} -c default_transaction_read_only=on -c statement_timeout=3000",
            ),
        )
        reachable, result = reader.lookup(pid)
        assert reachable is True
        assert result["status"] == "pending"
        assert result["pipeline"]["stage"] == "finalization"
        assert result["pipeline"]["generation"] == 2
        assert reader.has_active_work() == (True, True)
        with reader._connect() as connection:
            with connection.cursor() as cursor:
                with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
                    cursor.execute("UPDATE pipeline_job SET state='terminal'")
        with admin.cursor() as cursor:
            cursor.execute("UPDATE pipeline_job SET state='terminal'")
            cursor.execute("UPDATE extraction_run SET status='succeeded'")
        assert reader.has_active_work() == (True, False)
        assert reader.lookup(pid)[1]["status"] == "complete"
        # The ordered rollout forbids pre-migration startup; absence is unknown,
        # never an invented empty queue authorizing destructive replacement.
        with admin.cursor() as cursor:
            cursor.execute("DROP TABLE pipeline_job")
        assert reader.has_active_work() == (False, None)
        assert reader.lookup(pid) == (False, None)
    finally:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
        admin.close()
