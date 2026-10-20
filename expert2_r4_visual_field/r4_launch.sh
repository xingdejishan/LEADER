#!/bin/bash
# Launch one arm across N worker processes and wait. No nohup/setsid/systemd: the workers
# are plain children of this shell, so they die with it and can always be killed.
#   N=16 ARM=main TAG=main bash r4_launch.sh
set -u
cd /home/zhangcy/projects/r4-full-20261019-v1
N=${N:-16}
ARM=${ARM:-main}
TAG=${TAG:-$ARM}
EXTRA=${EXTRA:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/home/zhangcy/.conda/envs/leader/bin/python
LOGD=/data2/zhangcy/r4-full-20261019-v1/logs
mkdir -p "$LOGD"
echo "[launch] arm=$ARM tag=$TAG workers=$N extra='$EXTRA' host=$(hostname) at $(date -Is)"
uptime
pids=()
for ((w = 0; w < N; w++)); do
  nice -n 19 "$PY" r4_run.py --arm "$ARM" --of "$N" --worker "$w" --tag "$TAG" $EXTRA \
      > "$LOGD/$TAG.$w.out" 2>&1 &
  pids+=($!)
done
trap 'kill ${pids[@]} 2>/dev/null; exit 130' INT TERM
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[launch] all $N workers finished rc=$rc at $(date -Is)"
ls -1 /data2/zhangcy/r4-full-20261019-v1/results/"$TAG" | wc -l
