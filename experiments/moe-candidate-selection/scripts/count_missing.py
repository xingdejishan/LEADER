"""How many frames in the four test dates lack a cam5 image or a scan (read-only)."""
import glob
import os

ROOT = '/data2/dataset/NCLT'
for date in ('2012-02-12', '2012-02-19', '2012-03-31', '2012-05-26'):
    bins = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, date, 'velodyne_sync', '*.bin')))
    jpgs = set(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, date, 'cam5', '*.jpg')))
    missing = [n for n in bins if n not in jpgs]
    print('%-12s scans=%6d jpgs=%6d missing_jpg=%3d %s'
          % (date, len(bins), len(jpgs), len(missing), missing[:6]))
