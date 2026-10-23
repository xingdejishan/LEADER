"""Check the actual data roots the loader uses for the suspect frame (read-only)."""
import glob
import os

T = '1333216762828479'
DATE = '2012-03-31'
LEADER = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
for label, root in (('leader-tree', LEADER), ('dataset-tree', '/data2/dataset/NCLT')):
    for kind, sub, ext in (('scan', 'velodyne_sync', '.bin'), ('image', 'cam5', '.jpg')):
        p = os.path.join(root, DATE, sub, T + ext)
        print('%-13s %-6s exists=%-5s %s' % (label, kind, os.path.exists(p), p))

# coverage in the leader tree for the four dates
for date in ('2012-02-12', '2012-02-19', '2012-03-31', '2012-05-26'):
    bins = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(LEADER, date, 'velodyne_sync', '*.bin')))
    jpgs = set(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(LEADER, date, 'cam5', '*.jpg')))
    missing = [n for n in bins if n not in jpgs]
    print('%-12s scans=%6d jpgs=%6d missing_jpg=%3d %s'
          % (date, len(bins), len(jpgs), len(missing), missing[:6]))
