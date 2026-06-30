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
        self._startup_timeout_total: int = 0
        self._replacement_requests_total: int = 0
        self._last_health_status_code: Optional[int] = None
        self._last_health_checks: dict = {}
        self._last_health_reason: Optional[str] = None

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

    @property
    def startup_timeout_total(self) -> int:
        return self._startup_timeout_total

    @property
    def replacement_requests_total(self) -> int:
        return self._replacement_requests_total

    @property
    def last_health_status_code(self) -> Optional[int]:
        return self._last_health_status_code

    @property
    def last_health_checks(self) -> dict:
        return dict(self._last_health_checks)

    @property
    def last_health_reason(self) -> Optional[str]:
        return self._last_health_reason

    def _clear_health_snapshot(self) -> None:
        self._last_health_status_code = None
        self._last_health_checks = {}
        self._last_health_reason = None

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

    async def refresh_health_snapshot(self) -> bool:
        """Refresh cached backend health details without changing EC2 state."""
        if self._state not in (InstanceState.READY, InstanceState.BUSY) or not self._private_ip:
            return False
        return await self._check_health()

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
        self._clear_health_snapshot()
        self._startup_task = asyncio.create_task(self._poll_until_healthy())

    async def _poll_until_healthy(self) -> None:
        """Poll EC2 health endpoint until ready or timeout."""
        deadline = time.time() + settings.STARTUP_TIMEOUT_MINUTES * 60
        poll_interval = settings.HEALTH_POLL_INTERVAL_SECONDS
        start_requested = False
        replacement_attempts = 0

        while True:
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
            self._startup_timeout_total += 1
            can_wait_for_replacement = replacement_attempts < settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS
            replacement_requested = False

            if can_wait_for_replacement:
                try:
                    replacement_requested = self._ec2.mark_unhealthy()
                    if replacement_requested:
                        self._replacement_requests_total += 1
                except Exception as exc:
                    logger.error("Failed to request backend replacement after startup timeout: %s", exc)

            if replacement_requested:
                replacement_attempts += 1
                logger.warning(
                    "Waiting for ASG backend replacement attempt %d/%d",
                    replacement_attempts,
                    settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS,
                )
                self._state = InstanceState.STARTING
                self._private_ip = None
                self._ready_since = None
                self._clear_health_snapshot()
                start_requested = False
                deadline = time.time() + settings.STARTUP_TIMEOUT_MINUTES * 60
                continue

            try:
                self._ec2.stop_instance()
            except Exception as exc:
                logger.error("Failed to stop backend after terminal startup failure: %s", exc)
            break

        self._state = InstanceState.STOPPED
        self._private_ip = None
        self._ready_since = None
        self._clear_health_snapshot()

    async def _check_health(self) -> bool:
        """Hit the EC2 Flask health endpoint and validate backend availability.

        A solo Celery worker can be too busy to answer inspect/ping while it is
        processing a PDF. If the backend reports healthy dependencies plus a
        broker-unacked running task, treat it as available for submissions so
        queued work is reported honestly instead of as a startup failure.
        """
        if not self._private_ip:
            self._last_health_status_code = None
            self._last_health_checks = {}
            self._last_health_reason = "missing_private_ip"
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.ec2_base_url}/api/v1/health")
            self._last_health_status_code = resp.status_code

            payload = resp.json()
            if not isinstance(payload, dict):
                self._last_health_checks = {}
                self._last_health_reason = "invalid_health_payload"
                return False

            checks = payload.get("checks")
            if not isinstance(checks, dict):
                self._last_health_checks = {}
                self._last_health_reason = "missing_health_checks"
                return False

            self._last_health_checks = dict(checks)

            if checks.get("redis") != "ok":
                self._last_health_reason = "redis_not_ready"
                logger.debug("EC2 health not ready: redis=%r", checks.get("redis"))
                return False

            if checks.get("grobid") != "ok":
                self._last_health_reason = "grobid_not_ready"
                logger.debug("EC2 health not ready: grobid=%r", checks.get("grobid"))
                return False

            if resp.status_code != 200:
                self._last_health_reason = f"downstream_health_status_{resp.status_code}"
                logger.debug("EC2 health not ready: status=%s", resp.status_code)
                return False

            workers = checks.get("workers")
            if isinstance(workers, int) and workers > 0:
                self._last_health_reason = None
                return True

            fresh_active_runs = checks.get("fresh_active_runs")
            broker_unacked = checks.get("broker_unacked")
            if (
                isinstance(fresh_active_runs, int)
                and fresh_active_runs > 0
                and isinstance(broker_unacked, int)
                and broker_unacked > 0
                and checks.get("service") == "ok"
                and checks.get("worker_state") == "busy_or_unresponsive"
            ):
                self._last_health_reason = "worker_busy_or_unresponsive"
                logger.info(
                    "EC2 backend accepts submissions but worker inspect is not responsive: "
                    "workers=%r fresh_active_runs=%r broker_unacked=%r",
                    workers,
                    fresh_active_runs,
                    broker_unacked,
                )
                return True

            self._last_health_reason = "no_ready_workers"
            logger.debug("EC2 health not ready: workers=%r", workers)
            return False
        except Exception:
            self._last_health_status_code = None
            self._last_health_checks = {}
            self._last_health_reason = "downstream_unreachable"
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
                self._clear_health_snapshot()
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
                self._clear_health_snapshot()
            else:
                self._state = InstanceState.STOPPED
                self._ready_since = None
                self._clear_health_snapshot()
                logger.info("Synced: EC2 is stopped")
        except Exception as exc:
            logger.warning("Failed to sync EC2 state: %s", exc)
            self._state = InstanceState.STOPPED
            self._ready_since = None
            self._clear_health_snapshot()
