"""External stage deadline/ownership monitor, independent of the worker's GIL.

The worker holds a PostgreSQL session advisory lock. This process renews its
lease only while that lock and generation still belong to the stage. Its stdin
pipe is a private stop channel: EOF means the worker finished or died.
"""

import os
import select
import signal
import sys

from app import pipeline
from pdfx_observability import initialize, capture_failure


def main(process_id, generation, stage):
    parent = os.getppid()
    parent_fd = os.pidfd_open(parent)
    initialize(stage)
    announced = False
    while os.getppid() == parent:
        # Stop is checked before and after DB operations, so normal completion
        # cannot be mistaken for a lost owner while the parent exits the scope.
        if select.select([sys.stdin], [], [], 0)[0]:
            return
        try:
            with pipeline.live_lock(process_id) as missing_owner:
                if missing_owner is not None:
                    raise pipeline.OwnershipLost("Worker ownership connection was lost")
            pipeline.heartbeat(process_id, generation)
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
        if select.select([sys.stdin], [], [], 5)[0]:
            return


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
