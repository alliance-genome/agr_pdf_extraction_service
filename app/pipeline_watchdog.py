"""External stage deadline/ownership monitor, independent of the worker's GIL.

The worker holds a PostgreSQL session advisory lock. This process renews its
lease only while that lock and generation still belong to the stage. Its stdin
pipe is a private stop channel: EOF means the worker finished or died.
"""

import os
import select
import signal
import subprocess
import sys

from app import pipeline
from app.pipeline_health import touch
from config import Config
from pdfx_observability import initialize, capture_failure


def main(process_id, generation, stage):
    parent = os.getppid()
    parent_fd = os.pidfd_open(parent)
    self_fd = os.pidfd_open(os.getpid())
    sentinel = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "app.pipeline_sentinel",
            str(parent_fd),
            str(self_fd),
        ],
        stdin=sys.stdin,
        stdout=subprocess.PIPE,
        bufsize=0,
        pass_fds=(parent_fd, self_fd),
    )
    os.close(self_fd)
    sentinel_fd = os.pidfd_open(sentinel.pid)
    ready = select.select(
        [sentinel.stdout], [], [], Config.PIPELINE_WATCHDOG_STARTUP_SECONDS
    )[0]
    if not ready or sentinel.stdout.readline() != b"READY\n":
        signal.pidfd_send_signal(parent_fd, signal.SIGKILL)
        return
    sentinel.stdout.close()
    initialize(stage)
    announced = False
    while os.getppid() == parent:
        # Stop is checked before and after DB operations, so normal completion
        # cannot be mistaken for a lost owner while the parent exits the scope.
        if select.select([sys.stdin], [], [], 0)[0]:
            return
        try:
            if select.select([sentinel_fd], [], [], 0)[0]:
                raise RuntimeError("Stage watchdog sentinel exited unexpectedly")
            with pipeline.live_lock(process_id) as missing_owner:
                if missing_owner is not None:
                    raise pipeline.OwnershipLost("Worker ownership connection was lost")
            pipeline.heartbeat(process_id, generation)
            touch(stage)
            if not announced:
                print("READY", flush=True)
                announced = True
        except Exception as exc:
            if select.select([sys.stdin], [], [], 0)[0] or os.getppid() != parent:
                return
            try:
                if isinstance(exc, pipeline.PipelineDeadlineExceeded):
                    pipeline.fail(process_id, generation, exc)
                elif not isinstance(exc, pipeline.OwnershipLost):
                    capture_failure(
                        exc, process_id=process_id, stage=stage, generation=generation
                    )
            finally:
                # Kill only the exact parent which launched this stage monitor.
                # Finalizers are prefork children, so other jobs keep running.
                signal.pidfd_send_signal(parent_fd, signal.SIGKILL)
            return
        ready = select.select([sys.stdin, sentinel_fd], [], [], 5)[0]
        if sys.stdin in ready:
            return


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
