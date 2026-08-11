"""EC2 lifecycle state machine with idle timer and health polling."""

import asyncio
import enum
import logging
import time
from contextlib import suppress
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
        self._transition_lock = asyncio.Lock()
        self._startup_generation: int = 0
        self._startup_instance_id: Optional[str] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._active_jobs: int = 0
        self._stop_guard: Optional[Callable[[], bool]] = None
        self._replacement_guard: Optional[Callable[[], bool]] = None
        self._stop_events_total: int = 0
        self._stop_blocked_total: int = 0
        self._startup_timeout_total: int = 0
        self._replacement_requests_total: int = 0
        self._stale_monitor_exits_total: int = 0
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
    def stale_monitor_exits_total(self) -> int:
        return self._stale_monitor_exits_total

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

    def set_replacement_guard(self, guard: Callable[[], bool]) -> None:
        """Register the shared-active-work precondition for startup destruction."""
        self._replacement_guard = guard

    async def _shared_work_allows_replacement(self) -> bool:
        if self._replacement_guard is None:
            logger.error("No shared-work replacement guard is configured; refusing destructive action")
            return False
        try:
            return bool(await asyncio.to_thread(self._replacement_guard))
        except Exception as exc:
            logger.error("Shared-work replacement guard failed: %s", exc)
            return False

    def touch(self) -> None:
        """Reset the idle timer. Call on every incoming request."""
        self._last_activity = time.time()

    def job_started(self) -> None:
        self._active_jobs += 1
        self._state = InstanceState.BUSY

    def job_finished(self) -> None:
        self._active_jobs = max(0, self._active_jobs - 1)
        self.touch()
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
        async with self._transition_lock:
            if self._state in (InstanceState.READY, InstanceState.BUSY):
                return
            if self._startup_task and not self._startup_task.done():
                return
            self._begin_startup_locked()

    def _begin_startup_locked(self, instance_id: str | None = None) -> int:
        """Create the sole monitor for a new authoritative startup generation."""
        self._startup_generation += 1
        generation = self._startup_generation
        self._startup_instance_id = instance_id
        self._state = InstanceState.STARTING
        self._private_ip = None
        self._ready_since = None
        self._clear_health_snapshot()
        self._startup_task = asyncio.create_task(self._poll_until_healthy(generation))
        return generation

    def _owns_startup(self, generation: int) -> bool:
        return self._state == InstanceState.STARTING and self._startup_generation == generation

    async def _record_stale_monitor_exit(self, generation: int, reason: str) -> None:
        self._stale_monitor_exits_total += 1
        logger.info(
            "event=stale_startup_monitor_exit generation=%s current_generation=%s reason=%s",
            generation,
            self._startup_generation,
            reason,
        )

    async def _set_ready_if_owner(
        self,
        generation: int,
        private_ip: str,
        instance_id: str | None,
    ) -> bool:
        async with self._transition_lock:
            if not self._owns_startup(generation):
                return False
            current_task = asyncio.current_task()
            old_task = self._startup_task
            self._startup_generation += 1
            self._startup_instance_id = instance_id
            self._startup_task = None
            self._state = InstanceState.READY
            self._private_ip = private_ip
            self._ready_since = time.time()
            if old_task and old_task is not current_task and not old_task.done():
                old_task.cancel()
            self._start_idle_monitor()
            return True

    async def _poll_until_healthy(self, generation: int | None = None) -> None:
        """Poll EC2 health endpoint until ready or timeout."""
        current_task = asyncio.current_task()
        if generation is None:
            async with self._transition_lock:
                if self._startup_generation == 0:
                    self._startup_generation = 1
                generation = self._startup_generation
                self._state = InstanceState.STARTING
                self._startup_task = current_task

        deadline = time.time() + settings.STARTUP_TIMEOUT_MINUTES * 60
        poll_interval = settings.HEALTH_POLL_INTERVAL_SECONDS
        start_requested = False
        replacement_attempts = 0
        target_instance_id = self._startup_instance_id

        try:
            while True:
                while time.time() < deadline:
                    async with self._transition_lock:
                        if not self._owns_startup(generation):
                            await self._record_stale_monitor_exit(generation, "generation_superseded")
                            return
                    try:
                        ec2_state, ip, instance_id = await asyncio.to_thread(
                            self._ec2.get_instance_snapshot
                        )
                        async with self._transition_lock:
                            if not self._owns_startup(generation):
                                await self._record_stale_monitor_exit(generation, "snapshot_superseded")
                                return
                            if instance_id and target_instance_id is None:
                                target_instance_id = instance_id
                                self._startup_instance_id = instance_id
                            if ip:
                                self._private_ip = ip

                        if ec2_state == "stopped" and not start_requested:
                            logger.info("EC2 reached stopped during startup poll; issuing start request.")
                            await asyncio.to_thread(self._ec2.start_instance)
                            start_requested = True
                        if ec2_state == "running" and ip and await self._check_health(ip):
                            logger.info("EC2 instance healthy at %s", ip)
                            if await self._set_ready_if_owner(generation, ip, instance_id):
                                return
                            await self._record_stale_monitor_exit(generation, "healthy_but_superseded")
                            return
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.debug("Health poll error (expected during startup): %s", exc)

                    await asyncio.sleep(poll_interval)

                async with self._transition_lock:
                    if not self._owns_startup(generation):
                        await self._record_stale_monitor_exit(generation, "deadline_superseded")
                        return
                    self._startup_timeout_total += 1
                logger.error("EC2 startup timed out after %d minutes", settings.STARTUP_TIMEOUT_MINUTES)

                try:
                    ec2_state, ip, current_instance_id = await asyncio.to_thread(
                        self._ec2.get_instance_snapshot
                    )
                except Exception as exc:
                    logger.error("Failed to refresh backend identity after startup timeout: %s", exc)
                    ec2_state, ip, current_instance_id = "unknown", None, None

                async with self._transition_lock:
                    if not self._owns_startup(generation):
                        await self._record_stale_monitor_exit(generation, "timeout_snapshot_superseded")
                        return

                if target_instance_id and current_instance_id != target_instance_id:
                    await self._record_stale_monitor_exit(generation, "instance_identity_changed")
                    return
                if target_instance_id is None and current_instance_id:
                    target_instance_id = current_instance_id
                    async with self._transition_lock:
                        if self._owns_startup(generation):
                            self._startup_instance_id = current_instance_id

                if ec2_state == "running" and ip and await self._check_health(ip):
                    if await self._set_ready_if_owner(generation, ip, current_instance_id):
                        return
                    await self._record_stale_monitor_exit(generation, "timeout_health_superseded")
                    return

                # Re-read identity after the awaited health call. EC2Manager also
                # validates the exact target immediately before the AWS mutation.
                try:
                    _state, _ip, verified_instance_id = await asyncio.to_thread(
                        self._ec2.get_instance_snapshot
                    )
                except Exception as exc:
                    logger.error("Failed final backend identity check: %s", exc)
                    verified_instance_id = None

                async with self._transition_lock:
                    if not self._owns_startup(generation):
                        await self._record_stale_monitor_exit(generation, "final_check_superseded")
                        return
                if target_instance_id and verified_instance_id != target_instance_id:
                    await self._record_stale_monitor_exit(generation, "final_identity_changed")
                    return

                can_replace = (
                    target_instance_id is not None
                    and replacement_attempts < settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS
                )
                replacement_requested = False
                if can_replace:
                    if not await self._shared_work_allows_replacement():
                        logger.warning(
                            "Deferring backend replacement while shared running-work state is active or unavailable"
                        )
                        deadline = time.time() + max(1, settings.STARTUP_TIMEOUT_MINUTES * 60)
                        continue
                    async with self._transition_lock:
                        if not self._owns_startup(generation):
                            await self._record_stale_monitor_exit(generation, "replacement_guard_superseded")
                            return
                    try:
                        replacement_requested = await asyncio.to_thread(
                            self._ec2.mark_unhealthy,
                            target_instance_id,
                        )
                    except Exception as exc:
                        logger.error("Failed to request backend replacement after startup timeout: %s", exc)

                if replacement_requested:
                    async with self._transition_lock:
                        if not self._owns_startup(generation):
                            await self._record_stale_monitor_exit(generation, "replacement_superseded")
                            return
                        self._replacement_requests_total += 1
                        self._startup_instance_id = None
                        self._private_ip = None
                        self._ready_since = None
                        self._clear_health_snapshot()
                    replacement_attempts += 1
                    target_instance_id = None
                    start_requested = False
                    deadline = time.time() + settings.STARTUP_TIMEOUT_MINUTES * 60
                    logger.warning(
                        "Waiting for ASG backend replacement attempt %d/%d",
                        replacement_attempts,
                        settings.ASG_STARTUP_REPLACEMENT_ATTEMPTS,
                    )
                    continue

                if target_instance_id is None:
                    await self._record_stale_monitor_exit(generation, "destructive_target_unresolved")
                    return

                if not await self._shared_work_allows_replacement():
                    logger.warning(
                        "Deferring terminal startup stop while shared running-work state is active or unavailable"
                    )
                    deadline = time.time() + max(1, settings.STARTUP_TIMEOUT_MINUTES * 60)
                    continue
                async with self._transition_lock:
                    if not self._owns_startup(generation):
                        await self._record_stale_monitor_exit(generation, "stop_guard_superseded")
                        return

                try:
                    stopped = await asyncio.to_thread(
                        self._ec2.stop_instance,
                        target_instance_id,
                    )
                except Exception as exc:
                    logger.error("Failed to stop backend after terminal startup failure: %s", exc)
                    stopped = False

                async with self._transition_lock:
                    if self._owns_startup(generation) and stopped:
                        self._startup_generation += 1
                        self._startup_instance_id = None
                        self._state = InstanceState.STOPPED
                        self._private_ip = None
                        self._ready_since = None
                        self._clear_health_snapshot()
                return
        finally:
            async with self._transition_lock:
                if self._startup_task is current_task:
                    self._startup_task = None

    async def _check_health(self, private_ip: str | None = None) -> bool:
        """Hit the EC2 Flask health endpoint and validate backend availability.

        A solo Celery worker can be too busy to answer inspect/ping while it is
        processing a PDF. If the backend reports healthy dependencies plus a
        broker-unacked running task, treat it as available for submissions so
        queued work is reported honestly instead of as a startup failure.
        """
        health_ip = private_ip or self._private_ip
        if not health_ip:
            self._last_health_status_code = None
            self._last_health_checks = {}
            self._last_health_reason = "missing_private_ip"
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"http://{health_ip}:{settings.EC2_PORT}/api/v1/health")
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

            if "database" in checks and checks.get("database") != "ok":
                self._last_health_reason = "database_not_ready"
                logger.debug("EC2 health not ready: database=%r", checks.get("database"))
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
                and checks.get("worker_state") in {"busy", "busy_or_unresponsive"}
            ):
                self._last_health_reason = "worker_busy"
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
            with suppress(Exception):
                await self.refresh_health_snapshot()
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
                try:
                    ec2_state, ip, instance_id = await asyncio.to_thread(
                        self._ec2.get_instance_snapshot
                    )
                except Exception as exc:
                    logger.error("Failed to identify idle backend before stop: %s", exc)
                    continue
                if ec2_state != "running" or not instance_id:
                    continue
                if not ip:
                    continue
                try:
                    health_confirmed = await self._check_health(ip)
                except Exception:
                    health_confirmed = False
                if not health_confirmed:
                    logger.info(
                        "event=idle_stop_blocked reason=fresh_health_unavailable instance=%s",
                        instance_id,
                    )
                    self._stop_blocked_total += 1
                    continue
                # Health and AWS reads are await boundaries. Recheck process-local
                # work and current identity immediately before the exact-target stop.
                if self._state != InstanceState.READY:
                    continue
                if self._stop_guard:
                    try:
                        if not self._stop_guard():
                            self._stop_blocked_total += 1
                            continue
                    except Exception as exc:
                        logger.error("Stop guard callback failed during final recheck: %s", exc)
                        self._stop_blocked_total += 1
                        continue
                try:
                    _state, _ip, verified_instance_id = await asyncio.to_thread(
                        self._ec2.get_instance_snapshot
                    )
                except Exception as exc:
                    logger.error("Failed final idle backend identity check: %s", exc)
                    continue
                if verified_instance_id != instance_id:
                    logger.info(
                        "event=stale_idle_stop_exit expected_instance=%s current_instance=%s",
                        instance_id,
                        verified_instance_id,
                    )
                    continue
                logger.info(
                    "Idle timeout reached (%.0f seconds). Stopping EC2.",
                    self.idle_seconds,
                )
                try:
                    stopped = await asyncio.to_thread(self._ec2.stop_instance, instance_id)
                    if stopped:
                        self._stop_events_total += 1
                except Exception as exc:
                    logger.error("Failed to stop EC2: %s", exc)
                    stopped = False
                if not stopped:
                    continue
                self._state = InstanceState.STOPPED
                self._private_ip = None
                self._ready_since = None
                self._clear_health_snapshot()
                return

    async def sync_state_from_ec2(self) -> None:
        """Sync internal state with actual EC2 state. Call on proxy startup."""
        try:
            ec2_state, ip, instance_id = await asyncio.to_thread(self._ec2.get_instance_snapshot)
            if ec2_state == "running" and ip:
                if await self._check_health(ip):
                    async with self._transition_lock:
                        old_task = self._startup_task
                        self._startup_generation += 1
                        self._startup_instance_id = instance_id
                        self._startup_task = None
                        self._state = InstanceState.READY
                        self._private_ip = ip
                        self._ready_since = time.time()
                        if old_task and old_task is not asyncio.current_task() and not old_task.done():
                            old_task.cancel()
                        self._start_idle_monitor()
                    logger.info("Synced: EC2 is running and healthy at %s", ip)
                else:
                    async with self._transition_lock:
                        if not (self._startup_task and not self._startup_task.done()):
                            self._begin_startup_locked(instance_id)
                    logger.info("Synced: EC2 is running but not yet healthy")
            elif ec2_state in ("pending", "shutting-down", "stopping"):
                async with self._transition_lock:
                    if not (self._startup_task and not self._startup_task.done()):
                        self._begin_startup_locked(instance_id)
            else:
                async with self._transition_lock:
                    if self._startup_task and not self._startup_task.done():
                        return
                    self._startup_generation += 1
                    self._startup_instance_id = None
                    self._state = InstanceState.STOPPED
                    self._private_ip = None
                    self._ready_since = None
                    self._clear_health_snapshot()
                logger.info("Synced: EC2 is stopped")
        except Exception as exc:
            logger.warning("Failed to sync EC2 state: %s", exc)
            async with self._transition_lock:
                if not (self._startup_task and not self._startup_task.done()):
                    self._startup_generation += 1
                    self._startup_instance_id = None
                    self._state = InstanceState.STOPPED
                    self._private_ip = None
                    self._ready_since = None
                    self._clear_health_snapshot()
