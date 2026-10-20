import resource,shutil
from pathlib import Path

import torch


def budget(output):
    if shutil.disk_usage(output).free<10*1024**3:raise RuntimeError('covolume disk floor reached')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>10*1024**3:raise RuntimeError('covolume process memory limit reached')
    available=int([line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')][0].split()[1])*1024
    if available<32*1024**3:raise RuntimeError('covolume system memory floor reached')
    if sum(path.stat().st_size for path in Path(output).rglob('*') if path.is_file())>4*1024**3:raise RuntimeError('covolume output size limit reached')


def setup(output):
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    budget(output)
