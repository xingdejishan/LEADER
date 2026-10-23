"""Split the shard-1 ledger in two and relaunch the remaining work as two pipelined
processes (the 1/4 and 3/4 partitions together equal the old 1/2 partition)."""
import json
import os
import subprocess
import time

PROJ = '/home/zhangcy/projects/e2e-moe-20261020-v1'
PY = '/home/zhangcy/.conda/envs/leader/bin/python'
SRC = os.path.join(PROJ, 'results/full-shard1.jsonl')

rows = [json.loads(l) for l in open(SRC) if l.strip()]
even = [r for i, r in enumerate(rows) if i % 2 == 0]
odd = [r for i, r in enumerate(rows) if i % 2 == 1]
for name, part in (('fullb-shard1.jsonl', even), ('fullb-shard3.jsonl', odd)):
    path = os.path.join(PROJ, 'results', name)
    with open(path, 'w') as fh:
        for r in part:
            fh.write(json.dumps(r) + '\n')
    print('seeded %s with %d rows' % (name, len(part)))

# stop the serial run
subprocess.run(['pkill', '-u', 'zhangcy', '-f', 'e2e_online.py'], check=False)
time.sleep(8)

for shard in ('1/4', '3/4'):
    log = open(os.path.join(PROJ, 'results', 'fullb-shard%s.log' % shard[0]), 'w')
    proc = subprocess.Popen(
        [PY, 'e2e_online.py', '--keys-from', 'records', '--dates', '',
         '--shard', shard, '--refine-replicas', '4', '--out-prefix', 'results/fullb'],
        cwd=PROJ, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        start_new_session=True)
    print('launched shard %s pid=%d' % (shard, proc.pid))
time.sleep(5)
print(subprocess.run(['pgrep', '-u', 'zhangcy', '-f', 'e2e_online.py'],
                     capture_output=True, text=True).stdout.strip())
