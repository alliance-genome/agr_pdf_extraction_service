"""Tests for the EC2 lifecycle state machine."""

import time
import pytest
from unittest.mock import MagicMock

from app.state_machine import InstanceState, LifecycleManager


class TestLifecycleManager:
    def _make_manager(self, initial_state=InstanceState.STOPPED):
        ec2 = MagicMock()
        mgr = LifecycleManager(ec2)
        mgr._state = initial_state
        return mgr, ec2

    def test_initial_state_stopped(self):
        ec2 = MagicMock()
        mgr = LifecycleManager(ec2)
        assert mgr.state == InstanceState.STOPPED

    def test_touch_resets_idle_timer(self):
        mgr, _ = self._make_manager()
        mgr._last_activity = 0
        mgr.touch()
        assert mgr._last_activity > 0

    def test_idle_seconds(self):
        mgr, _ = self._make_manager()
        mgr._last_activity = time.time() - 120
        assert mgr.idle_seconds >= 119

    def test_job_started_sets_busy(self):
        mgr, _ = self._make_manager(InstanceState.READY)
        mgr.job_started()
        assert mgr.state == InstanceState.BUSY
        assert mgr.active_jobs == 1

    def test_job_finished_returns_to_ready(self):
        mgr, _ = self._make_manager(InstanceState.BUSY)
        mgr._active_jobs = 1
        mgr.job_finished()
        assert mgr.state == InstanceState.READY
        assert mgr.active_jobs == 0

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

    @pytest.mark.asyncio
    async def test_ensure_running_noop_when_ready(self):
        mgr, ec2 = self._make_manager(InstanceState.READY)
        await mgr.ensure_running()
        ec2.start_instance.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_running_noop_when_starting(self):
        mgr, ec2 = self._make_manager(InstanceState.STARTING)
        await mgr.ensure_running()
        ec2.start_instance.assert_not_called()
