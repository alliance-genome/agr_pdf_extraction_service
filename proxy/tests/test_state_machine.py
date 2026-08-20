"""Tests for the EC2 lifecycle state machine."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.state_machine import InstanceState, LifecycleManager


class TestLifecycleManager:
    def _make_manager(self, initial_state=InstanceState.STOPPED):
        ec2 = MagicMock()
        mgr = LifecycleManager(ec2)
        mgr.set_replacement_guard(lambda: True)
        mgr._state = initial_state
        return mgr, ec2

    def test_initial_state_stopped(self):
        ec2 = MagicMock()
        mgr = LifecycleManager(ec2)
        assert mgr.state == InstanceState.STOPPED
        assert mgr.backend_work_observed is False

    def test_touch_resets_idle_timer(self):
        mgr, _ = self._make_manager()
        mgr._last_activity = 0
        mgr.touch()
        assert mgr._last_activity > 0
        assert mgr.backend_work_observed is False

    def test_request_activity_does_not_reset_backend_work_timer(self):
        mgr, _ = self._make_manager()
        mgr._last_backend_work = time.time() - 120
        mgr.record_backend_work()
        mgr._last_backend_work = time.time() - 120

        mgr.touch()

        assert mgr.idle_seconds < 1
        assert mgr.backend_work_idle_seconds >= 119
        assert mgr.backend_work_observed is True

    def test_idle_seconds(self):
        mgr, _ = self._make_manager()
        mgr._last_activity = time.time() - 120
        assert mgr.idle_seconds >= 119

    def test_job_started_sets_busy(self):
        mgr, _ = self._make_manager(InstanceState.READY)
        mgr.job_started()
        assert mgr.state == InstanceState.BUSY
        assert mgr.active_jobs == 1
        assert mgr.backend_work_observed is True

    def test_job_finished_returns_to_ready(self):
        mgr, _ = self._make_manager(InstanceState.BUSY)
        mgr._active_jobs = 1
        mgr._last_activity = 0
        mgr.job_finished()
        assert mgr.state == InstanceState.READY
        assert mgr.active_jobs == 0
        assert mgr._last_activity > 0

    def test_job_finished_stays_busy_with_remaining_jobs(self):
        mgr, _ = self._make_manager(InstanceState.BUSY)
        mgr._active_jobs = 2
        mgr.job_finished()
        assert mgr.state == InstanceState.BUSY
        assert mgr.active_jobs == 1

    def test_job_finished_clamps_at_zero(self):
        mgr, _ = self._make_manager(InstanceState.READY)
        mgr._active_jobs = 0
        mgr.job_finished()
        assert mgr.active_jobs == 0

    def test_ec2_base_url(self):
        mgr, _ = self._make_manager()
        mgr._private_ip = "172.31.1.100"
        assert mgr.ec2_base_url == "http://172.31.1.100:5000"

    def test_refresh_health_snapshot_only_when_backend_ready(self):
        mgr, _ = self._make_manager(InstanceState.STARTING)
        mgr._private_ip = "10.0.0.5"
        mgr._check_health = AsyncMock(return_value=True)

        assert asyncio.run(mgr.refresh_health_snapshot()) is False
        mgr._check_health.assert_not_awaited()

        mgr._state = InstanceState.READY
        assert asyncio.run(mgr.refresh_health_snapshot()) is True
        mgr._check_health.assert_awaited_once()

    def test_ensure_running_noop_when_ready(self):
        mgr, ec2 = self._make_manager(InstanceState.READY)
        asyncio.run(mgr.ensure_running())
        ec2.start_instance.assert_not_called()

    def test_ensure_running_noop_when_starting(self):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        asyncio.run(mgr.ensure_running())
        ec2.start_instance.assert_not_called()

    def test_26_concurrent_ensure_running_calls_share_one_startup_monitor(self, monkeypatch):
        mgr, ec2 = self._make_manager()
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        mgr._check_health = AsyncMock(return_value=True)
        mgr._start_idle_monitor = MagicMock()

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr("app.state_machine.asyncio.sleep", _no_sleep)

        async def _run():
            await asyncio.gather(*(mgr.ensure_running() for _ in range(26)))
            startup_task = mgr._startup_task
            assert startup_task is not None
            await startup_task
            return startup_task

        startup_task = asyncio.run(_run())

        assert mgr.state == InstanceState.READY
        assert ec2.get_instance_snapshot.call_count == 1
        assert mgr._startup_task is None
        assert startup_task.done()
        ec2.start_instance.assert_not_called()
        ec2.mark_unhealthy.assert_not_called()

    def test_concurrent_ensure_and_sync_reuse_authoritative_monitor(self, monkeypatch):
        mgr, ec2 = self._make_manager()
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        mgr._check_health = AsyncMock(return_value=True)
        mgr._start_idle_monitor = MagicMock()

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr("app.state_machine.asyncio.sleep", _no_sleep)

        async def _run():
            await asyncio.gather(mgr.ensure_running(), mgr.sync_state_from_ec2())
            if mgr._startup_task:
                await mgr._startup_task

        asyncio.run(_run())

        assert mgr.state == InstanceState.READY
        assert ec2.mark_unhealthy.call_count == 0
        assert mgr._startup_task is None

    def test_stale_monitor_exits_after_new_generation_is_ready(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        mgr._startup_generation = 2
        mgr._startup_instance_id = "i-current"
        mgr._state = InstanceState.READY
        mgr._private_ip = "10.0.0.5"

        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 0)

        asyncio.run(mgr._poll_until_healthy(1))

        assert mgr.state == InstanceState.READY
        assert mgr.private_ip == "10.0.0.5"
        ec2.mark_unhealthy.assert_not_called()
        ec2.stop_instance.assert_not_called()

    def test_timeout_cannot_replace_a_different_current_instance(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        mgr._startup_generation = 1
        mgr._startup_instance_id = "i-old"
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.9", "i-new")
        mgr._check_health = AsyncMock(return_value=False)

        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 0)
        monkeypatch.setattr("app.state_machine.settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS", 1)

        asyncio.run(mgr._poll_until_healthy(1))

        ec2.mark_unhealthy.assert_not_called()
        ec2.stop_instance.assert_not_called()

    def test_timeout_without_exact_instance_never_performs_destructive_action(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        mgr._startup_generation = 1
        ec2.get_instance_snapshot.return_value = ("unknown", None, None)

        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 0)
        monkeypatch.setattr("app.state_machine.settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS", 1)

        asyncio.run(mgr._poll_until_healthy(1))

        ec2.mark_unhealthy.assert_not_called()
        ec2.stop_instance.assert_not_called()

    def test_poll_until_healthy_starts_after_stopping_transitions_to_stopped(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        ec2.get_instance_snapshot.side_effect = [
            ("stopping", None, "i-current"),
            ("stopped", None, "i-current"),
            ("pending", None, "i-current"),
            ("running", "10.0.0.5", "i-current"),
        ]
        mgr._check_health = AsyncMock(return_value=True)
        mgr._start_idle_monitor = MagicMock()

        async def _no_sleep(_):
            return None

        monkeypatch.setattr("app.state_machine.asyncio.sleep", _no_sleep)

        asyncio.run(mgr._poll_until_healthy())

        ec2.start_instance.assert_called_once()
        assert mgr.state == InstanceState.READY
        assert mgr.private_ip == "10.0.0.5"

    def test_poll_until_healthy_stops_backend_after_terminal_timeout(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        ec2.mark_unhealthy.return_value = True
        mgr._check_health = AsyncMock(return_value=False)

        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 0)
        monkeypatch.setattr("app.state_machine.settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS", 0)

        asyncio.run(mgr._poll_until_healthy())

        ec2.mark_unhealthy.assert_not_called()
        ec2.stop_instance.assert_called_once()
        assert mgr.state == InstanceState.STOPPED
        assert mgr.startup_timeout_total == 1
        assert mgr.replacement_requests_total == 0

    def test_timeout_defers_and_rechecks_when_shared_work_is_active(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        mgr._startup_generation = 1
        mgr._startup_instance_id = "i-current"
        replacement_guard = MagicMock(side_effect=[False, asyncio.CancelledError()])
        mgr.set_replacement_guard(replacement_guard)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        ec2.mark_unhealthy.return_value = True
        mgr._check_health = AsyncMock(return_value=False)

        clock = {"now": 0.0}

        def _advancing_time():
            clock["now"] += 1.0
            return clock["now"]

        monkeypatch.setattr("app.state_machine.time.time", _advancing_time)
        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 0)
        monkeypatch.setattr("app.state_machine.settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS", 1)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mgr._poll_until_healthy(1))

        assert replacement_guard.call_count == 2
        ec2.mark_unhealthy.assert_not_called()
        ec2.stop_instance.assert_not_called()
        assert mgr.stale_monitor_exits_total == 0

    def test_terminal_startup_stop_defers_instead_of_abandoning_monitor(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        mgr._startup_generation = 1
        mgr._startup_instance_id = "i-current"
        replacement_guard = MagicMock(side_effect=[False, asyncio.CancelledError()])
        mgr.set_replacement_guard(replacement_guard)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        mgr._check_health = AsyncMock(return_value=False)

        clock = {"now": 0.0}

        def _advancing_time():
            clock["now"] += 1.0
            return clock["now"]

        monkeypatch.setattr("app.state_machine.time.time", _advancing_time)
        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 0)
        monkeypatch.setattr("app.state_machine.settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS", 0)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mgr._poll_until_healthy(1))

        assert replacement_guard.call_count == 2
        ec2.mark_unhealthy.assert_not_called()
        ec2.stop_instance.assert_not_called()
        assert mgr.stale_monitor_exits_total == 0

    def test_timeout_health_recheck_prevents_asg_replacement(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        ec2.mark_unhealthy.return_value = True
        mgr._check_health = AsyncMock(return_value=True)
        mgr._start_idle_monitor = MagicMock()

        times = iter([0.0, 2.0, 2.0, 2.1, 2.2])
        monkeypatch.setattr("app.state_machine.time.time", lambda: next(times, 2.2))
        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 1 / 60)
        monkeypatch.setattr("app.state_machine.settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS", 1)

        asyncio.run(mgr._poll_until_healthy())

        ec2.mark_unhealthy.assert_not_called()
        assert mgr.state == InstanceState.READY
        assert mgr.startup_timeout_total == 1
        assert mgr.replacement_requests_total == 0
        ec2.stop_instance.assert_not_called()

    def test_poll_until_healthy_stops_backend_after_exhausted_replacement(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        ec2.mark_unhealthy.return_value = True
        mgr._check_health = AsyncMock(return_value=False)

        fake_time = {"now": -1.0}

        def _advancing_time():
            fake_time["now"] += 1.0
            return fake_time["now"]

        monkeypatch.setattr("app.state_machine.time.time", _advancing_time)
        monkeypatch.setattr("app.state_machine.settings.STARTUP_TIMEOUT_MINUTES", 1 / 60)
        monkeypatch.setattr("app.state_machine.settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS", 1)

        asyncio.run(mgr._poll_until_healthy())

        ec2.mark_unhealthy.assert_called_once()
        ec2.stop_instance.assert_called_once()
        assert mgr.state == InstanceState.STOPPED
        assert mgr.startup_timeout_total == 2
        assert mgr.replacement_requests_total == 1

    def test_check_health_requires_active_workers(self, monkeypatch):
        mgr, _ = self._make_manager()
        mgr._private_ip = "10.0.0.5"

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "checks": {"grobid": "ok", "redis": "ok", "workers": 0},
                }

        class _Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, _url):
                return _Resp()

        monkeypatch.setattr("app.state_machine.httpx.AsyncClient", _Client)
        assert asyncio.run(mgr._check_health()) is False
        assert mgr.last_health_reason == "no_ready_workers"

    def test_check_health_rejects_unready_database(self, monkeypatch):
        mgr, _ = self._make_manager()
        mgr._private_ip = "10.0.0.5"

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "unhealthy",
                    "checks": {
                        "grobid": "ok",
                        "redis": "ok",
                        "database": "unavailable",
                        "workers": 1,
                    },
                }

        class _Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, _url):
                return _Resp()

        monkeypatch.setattr("app.state_machine.httpx.AsyncClient", _Client)
        assert asyncio.run(mgr._check_health()) is False
        assert mgr.last_health_reason == "database_not_ready"

    def test_check_health_accepts_busy_solo_worker(self, monkeypatch):
        mgr, _ = self._make_manager()
        mgr._private_ip = "10.0.0.5"

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "busy",
                    "checks": {
                        "service": "ok",
                        "grobid": "ok",
                        "redis": "ok",
                        "workers": 0,
                        "active_runs": 1,
                        "fresh_active_runs": 1,
                        "broker_unacked": 1,
                        "worker_state": "busy",
                    },
                }

        class _Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, _url):
                return _Resp()

        monkeypatch.setattr("app.state_machine.httpx.AsyncClient", _Client)
        assert asyncio.run(mgr._check_health()) is True
        assert mgr.last_health_reason == "worker_busy"
        assert mgr.last_health_checks["broker_unacked"] == 1

    def test_check_health_rejects_stale_running_row_without_unacked_task(self, monkeypatch):
        mgr, _ = self._make_manager()
        mgr._private_ip = "10.0.0.5"

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "unhealthy",
                    "checks": {
                        "service": "ok",
                        "grobid": "ok",
                        "redis": "ok",
                        "workers": 0,
                        "active_runs": 1,
                        "fresh_active_runs": 1,
                        "broker_unacked": 0,
                    },
                }

        class _Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, _url):
                return _Resp()

        monkeypatch.setattr("app.state_machine.httpx.AsyncClient", _Client)
        assert asyncio.run(mgr._check_health()) is False
        assert mgr.last_health_reason == "no_ready_workers"

    def test_sync_stopped_clears_cached_health_snapshot(self):
        mgr, ec2 = self._make_manager(InstanceState.READY)
        mgr._private_ip = "10.0.0.5"
        mgr._last_health_status_code = 200
        mgr._last_health_reason = "worker_busy_or_unresponsive"
        mgr._last_health_checks = {
            "grobid": "ok",
            "redis": "ok",
            "fresh_active_runs": 1,
            "broker_unacked": 1,
            "worker_state": "busy_or_unresponsive",
        }
        ec2.get_instance_snapshot.return_value = ("stopped", None, None)

        asyncio.run(mgr.sync_state_from_ec2())

        assert mgr.state == InstanceState.STOPPED
        assert mgr.private_ip is None
        assert mgr.last_health_status_code is None
        assert mgr.last_health_reason is None
        assert mgr.last_health_checks == {}

    def test_check_health_passes_with_workers_and_dependencies_ok(self, monkeypatch):
        mgr, _ = self._make_manager()
        mgr._private_ip = "10.0.0.5"

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "checks": {"grobid": "ok", "redis": "ok", "workers": 1},
                }

        class _Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, _url):
                return _Resp()

        monkeypatch.setattr("app.state_machine.httpx.AsyncClient", _Client)
        assert asyncio.run(mgr._check_health()) is True

    def test_idle_stop_names_and_revalidates_exact_instance(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.READY)
        mgr._private_ip = "10.0.0.5"
        mgr._ready_since = 0
        mgr._last_activity = 0
        mgr.set_stop_guard(lambda: True)
        mgr._check_health = AsyncMock(return_value=True)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")
        ec2.stop_instance.return_value = True

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr("app.state_machine.asyncio.sleep", _no_sleep)
        monkeypatch.setattr("app.state_machine.settings.IDLE_TIMEOUT_MINUTES", 0)
        monkeypatch.setattr("app.state_machine.settings.MIN_UPTIME_MINUTES", 0)

        asyncio.run(mgr._idle_monitor())

        ec2.stop_instance.assert_called_once_with("i-current")
        assert ec2.get_instance_snapshot.call_count == 2
        assert mgr.state == InstanceState.STOPPED

    def test_idle_stop_requires_fresh_application_health(self, monkeypatch):
        mgr, ec2 = self._make_manager(InstanceState.READY)
        mgr._private_ip = "10.0.0.5"
        mgr._ready_since = 0
        mgr._last_activity = 0
        mgr.set_stop_guard(lambda: True)
        mgr._check_health = AsyncMock(return_value=False)
        ec2.get_instance_snapshot.return_value = ("running", "10.0.0.5", "i-current")

        sleep_calls = 0

        async def _one_iteration(_seconds):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr("app.state_machine.asyncio.sleep", _one_iteration)
        monkeypatch.setattr("app.state_machine.settings.IDLE_TIMEOUT_MINUTES", 0)
        monkeypatch.setattr("app.state_machine.settings.MIN_UPTIME_MINUTES", 0)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mgr._idle_monitor())

        ec2.stop_instance.assert_not_called()
        assert mgr.stop_blocked_total >= 1
