"""Process liveness independent of a busy solo worker's Celery control replies."""

from pathlib import Path
import threading
import time
from config import Config


def path(role):
    return Path(Config.PIPELINE_HEALTH_DIRECTORY) / f"{role}.ready"


def touch(role):
    target = path(role)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()


def fresh(role):
    try:
        return (
            time.time() - path(role).stat().st_mtime
            <= Config.PIPELINE_HEALTH_MAX_AGE_SECONDS
        )
    except FileNotFoundError:
        return False


def start(role):
    def beat():
        while True:
            touch(role)
            time.sleep(5)

    threading.Thread(target=beat, daemon=True, name=f"{role}-health").start()
