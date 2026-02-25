"""EC2 lifecycle state machine with idle timer and health polling."""

import asyncio
import enum
import logging
import time
from typing import Callable, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class InstanceState(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"


class LifecycleManager:
    """Tracks EC2 state, manages health polling and idle shutdown."""

    def __init__(self, ec2_manager):
        self._ec2 = ec2_manager
        self._state = InstanceState.STOPPED
        self._private_ip: Optional[str] = None
        self._last_activity: float = time.time()
        self._ready_since: Optional[float] = None
        self._startup_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._active_jobs: int = 0
        self._stop_guard: Optional[Callable[[], bool]] = None
        self._stop_events_total: int = 0
        self._stop_blocked_total: int = 0

    @property
    def state(self) -> InstanceState:
        return self._state

    @property
    def private_ip(self) -> Optional[str]:
        return self._private_ip

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_activity

    @property
    def active_jobs(self) -> int:
        return self._active_jobs

    @property
    def ready_seconds(self) -> float:
        if self._ready_since is None:
            return 0.0
        return max(0.0, time.time() - self._ready_since)

    @property
    def stop_events_total(self) -> int:
        return self._stop_events_total

    @property
    def stop_blocked_total(self) -> int:
        return self._stop_blocked_total

    def set_stop_guard(self, guard: Callable[[], bool]) -> None:
        """Register callback that must return True before EC2 can stop."""
        self._stop_guard = guard

    def touch(self) -> None:
        """Reset the idle timer. Call on every incoming request."""
        self._last_activity = time.time()

    def job_started(self) -> None:
        self._active_jobs += 1
        self._state = InstanceState.BUSY

    def job_finished(self) -> None:
        self._active_jobs = max(0, self._active_jobs - 1)
        if self._active_jobs == 0 and self._state == InstanceState.BUSY:
            self._state = InstanceState.READY
            if self._ready_since is None:
                self._ready_since = time.time()

    @property
    def ec2_base_url(self) -> str:
        return f"http://{self._private_ip}:{settings.EC2_PORT}"

    async def ensure_running(self) -> None:
        """Start EC2 if stopped. Idempotent if already starting/running."""
        if self._state in (InstanceState.READY, InstanceState.BUSY):
            return
        if self._state == InstanceState.STARTING:
            return  # already booting

        # Check actual EC2 state first
        ec2_state, ip = self._ec2.get_instance_state()
        if ec2_state == "running" and ip:
            self._private_ip = ip
            if await self._check_health():
                self._state = InstanceState.READY
                if self._ready_since is None:
                    self._ready_since = time.time()
                self._start_idle_monitor()
                return

        if ec2_state == "stopped":
            self._ec2.start_instance()

        self._state = InstanceState.STARTING
        self._startup_task = asyncio.create_task(self._poll_until_healthy())

    async def _poll_until_healthy(self) -> None:
        """Poll EC2 health endpoint until ready or timeout."""
        deadline = time.time() + settings.STARTUP_TIMEOUT_MINUTES * 60
        poll_interval = settings.HEALTH_POLL_INTERVAL_SECONDS
        start_requested = False

        while time.time() < deadline:
            try:
                ec2_state, ip = self._ec2.get_instance_state()
                if ec2_state == "stopped" and not start_requested:
                    # If the first observed state was "stopping", issue start once
                    # after AWS transitions to fully "stopped".
                    logger.info("EC2 reached stopped during startup poll; issuing start request.")
                    self._ec2.start_instance()
                    start_requested = True
                if ip:
                    self._private_ip = ip
                if ec2_state == "running" and ip:
                    if await self._check_health():
                        logger.info("EC2 instance healthy at %s", ip)
                        self._state = InstanceState.READY
                        self._ready_since = time.time()
                        self._start_idle_monitor()
                        return
            except Exception as exc:
                logger.debug("Health poll error (expected during startup): %s", exc)

            await asyncio.sleep(poll_interval)

        logger.error("EC2 startup timed out after %d minutes", settings.STARTUP_TIMEOUT_MINUTES)
        self._state = InstanceState.STOPPED
        self._private_ip = None
        self._ready_since = None

    async def _check_health(self) -> bool:
        """Hit the EC2 Flask health endpoint and validate worker readiness."""
        if not self._private_ip:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.ec2_base_url}/api/v1/health")
                if resp.status_code != 200:
                    return False

                payload = resp.json()
                if not isinstance(payload, dict):
                    return False

                checks = payload.get("checks")
                if not isinstance(checks, dict):
                    return False

                workers = checks.get("workers")
                if not isinstance(workers, int) or workers <= 0:
                    logger.debug("EC2 health not ready: workers=%r", workers)
                    return False

                if checks.get("redis") != "ok":
                    logger.debug("EC2 health not ready: redis=%r", checks.get("redis"))
                    return False

                if checks.get("grobid") != "ok":
                    logger.debug("EC2 health not ready: grobid=%r", checks.get("grobid"))
                    return False

                return True
        except Exception:
            return False

    def _start_idle_monitor(self) -> None:
        """Start the background idle timer."""
        if self._idle_task and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_monitor())

    async def _idle_monitor(self) -> None:
        """Periodically check idle time and stop EC2 when threshold reached."""
        timeout = settings.IDLE_TIMEOUT_MINUTES * 60
        min_uptime_seconds = settings.MIN_UPTIME_MINUTES * 60
        while True:
            await asyncio.sleep(60)  # check every minute
            if self._state not in (InstanceState.READY, InstanceState.BUSY):
                return
            if settings.ALWAYS_ON_MODE:
                continue
            if self._state == InstanceState.BUSY:
                continue  # don't stop while jobs are running
            if self._ready_since and (time.time() - self._ready_since) < min_uptime_seconds:
                continue
            if self._stop_guard:
                try:
                    can_stop = self._stop_guard()
                except Exception as exc:
                    logger.error("Stop guard callback failed: %s", exc)
                    can_stop = False
                if not can_stop:
                    self._stop_blocked_total += 1
                    continue
            if self.idle_seconds >= timeout:
                logger.info(
                    "Idle timeout reached (%.0f seconds). Stopping EC2.",
                    self.idle_seconds,
                )
                try:
                    self._ec2.stop_instance()
                    self._stop_events_total += 1
                except Exception as exc:
                    logger.error("Failed to stop EC2: %s", exc)
                self._state = InstanceState.STOPPED
                self._private_ip = None
                self._ready_since = None
                return

    async def sync_state_from_ec2(self) -> None:
        """Sync internal state with actual EC2 state. Call on proxy startup."""
        try:
            ec2_state, ip = self._ec2.get_instance_state()
            self._private_ip = ip
            if ec2_state == "running" and ip:
                if await self._check_health():
                    self._state = InstanceState.READY
                    self._ready_since = time.time()
                    self._start_idle_monitor()
                    logger.info("Synced: EC2 is running and healthy at %s", ip)
                else:
                    self._state = InstanceState.STARTING
                    self._startup_task = asyncio.create_task(self._poll_until_healthy())
                    logger.info("Synced: EC2 is running but not yet healthy")
            elif ec2_state in ("pending", "shutting-down", "stopping"):
                self._state = InstanceState.STARTING
                self._startup_task = asyncio.create_task(self._poll_until_healthy())
                self._ready_since = None
            else:
                self._state = InstanceState.STOPPED
                self._ready_since = None
                logger.info("Synced: EC2 is stopped")
        except Exception as exc:
            logger.warning("Failed to sync EC2 state: %s", exc)
            self._state = InstanceState.STOPPED
            self._ready_since = None
