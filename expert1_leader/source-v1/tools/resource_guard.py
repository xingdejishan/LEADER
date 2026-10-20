import os
import shutil
from pathlib import Path


GIB = 1024 ** 3


class DiskSpaceError(RuntimeError):
    pass


class DiskGuard:
    def __init__(self, path, minimum_free_bytes):
        self.path = Path(path).resolve(strict=True)
        self.minimum_free_bytes = int(minimum_free_bytes)
        if self.minimum_free_bytes < 0:
            raise ValueError("disk free-space floor must be nonnegative")

    def available_bytes(self):
        return int(shutil.disk_usage(str(self.path)).free)

    def require(self, upcoming_write_bytes=0):
        upcoming = int(upcoming_write_bytes)
        if upcoming < 0:
            raise ValueError("upcoming write budget must be nonnegative")
        available = self.available_bytes()
        required = self.minimum_free_bytes + upcoming
        if available < required:
            raise DiskSpaceError(
                "filesystem free-space guard stopped the run: available={} required={}".format(
                    available, required))
        return available


def inherited_disk_guard():
    path = os.environ.get("LEADER_DISK_GUARD_PATH")
    floor = os.environ.get("LEADER_DISK_MIN_FREE_BYTES")
    if path is None and floor is None:
        return None
    if not path or not floor:
        raise ValueError("incomplete inherited LEADER disk guard")
    return DiskGuard(path, int(floor))
