"""Dedicated supervision of the lease watchdog, outside the worker GIL.

Both monitors inherit the worker-owned stop pipe. EOF takes precedence over
process death, so normal teardown cannot kill a worker finishing its stage.
"""

import os
import select
import signal
import sys


def main(worker_fd, watchdog_fd):
    try:
        try:
            print("READY", flush=True)
        except BrokenPipeError:
            # The watchdog may have died before the startup handshake.
            pass
        ready = select.select([sys.stdin, watchdog_fd, worker_fd], [], [])[0]
        if sys.stdin in ready or worker_fd in ready:
            return
        # Recheck the stop channel in case normal completion raced with select.
        if not select.select([sys.stdin], [], [], 0)[0]:
            signal.pidfd_send_signal(worker_fd, signal.SIGKILL)
    finally:
        os.close(watchdog_fd)
        os.close(worker_fd)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
