import os
from pathlib import Path
import shutil
import sys
import time


class ResourceBudget:
    def __init__(self, output, seconds=600, memory_gib=8):
        import resource
        if sys.prefix != '/home/zhangcy/.conda/envs/leader':
            raise RuntimeError('execute only inside the existing user leader environment')
        self.output = Path(output).resolve()
        allowed = Path('/home/zhangcy/projects/leader-ray-bundle-20260928').resolve()
        if self.output != allowed and allowed not in self.output.parents:
            raise ValueError('outputs must remain in the isolated ray-bundle project')
        if not allowed.is_dir() or allowed.stat().st_uid != os.getuid():
            raise ValueError('private project owner mismatch')
        self.start = time.monotonic()
        self.seconds = int(seconds)
        self.memory = int(memory_gib * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (self.memory, self.memory))
        resource.setrlimit(resource.RLIMIT_CPU, (self.seconds, self.seconds + 5))
        os.umask(0o077)
        self.check()

    def check(self):
        import resource
        root = Path('/home/zhangcy/projects/leader-ray-bundle-20260928')
        if shutil.disk_usage(root).free < 3 * 2**30:
            raise RuntimeError('disk floor reached')
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(memory['MemAvailable'].split()[0]) * 1024 < 32 * 2**30:
            raise RuntimeError('system available memory floor reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > self.memory * .8:
            raise RuntimeError('process RSS floor reached')
        if time.monotonic() - self.start > self.seconds:
            raise TimeoutError('wall time budget exhausted')
