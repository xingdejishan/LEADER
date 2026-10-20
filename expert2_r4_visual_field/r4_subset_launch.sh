#!/bin/bash
# Upper-bound arm (truth trajectory tangent) and the pre-revision u estimator, both on the
# fixed 100-frame subset, 8 workers each. Plain children of this shell; killable.
set -u
cd /home/zhangcy/projects/r4-full-20261019-v1
IDX=$(cat /data2/zhangcy/r4-full-20261019-v1/subset100.txt)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/home/zhangcy/.conda/envs/leader/bin/python
LOGD=/data2/zhangcy/r4-full-20261019-v1/logs
mkdir -p "$LOGD"
echo "[subset] frames=$(echo "$IDX" | tr ',' '\n' | wc -l) at $(date -Is)"
uptime
pids=()
for ARM in truth old; do
  for ((w = 0; w < 8; w++)); do
    nice -n 19 "$PY" r4_run.py --arm "$ARM" --of 8 --worker "$w" --tag "$ARM" \
        --indices "$IDX" > "$LOGD/$ARM.$w.out" 2>&1 &
    pids+=($!)
  done
done
trap 'kill ${pids[@]} 2>/dev/null; exit 130' INT TERM
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[subset] done rc=$rc at $(date -Is)"
echo "truth: $(ls -1 /data2/zhangcy/r4-full-20261019-v1/results/truth | wc -l)  old: $(ls -1 /data2/zhangcy/r4-full-20261019-v1/results/old | wc -l)"
