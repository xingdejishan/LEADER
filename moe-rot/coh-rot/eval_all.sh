#!/bin/bash
# Evaluate every arm twice, sequentially, on one GPU. One process at a time.
#   eval_all.sh <gpu>
set -uo pipefail

GPU=${1:?gpu}
SHM=/dev/shm/cohrot-20261010
PY=/home/zhangcy/.conda/envs/leader/bin/python
OUT="$SHM/eval"
mkdir -p "$OUT"
cd /home/zhangcy/projects/coh-rot-led-20261010 || exit 1

echo "=== full evaluation chain on GPU $GPU ==="
date -Is
for spec in \
    "base:$SHM/base/official_l0.pt:0" \
    "A:$SHM/run-A/armA.pt:1" \
    "A:$SHM/run-A/armA.pt:2" \
    "B:$SHM/run-B/armB.pt:1" \
    "B:$SHM/run-B/armB.pt:2" \
    "C:$SHM/run-C/armC.pt:1" \
    "C:$SHM/run-C/armC.pt:2" \
; do
    TAG=${spec%%:*}
    rest=${spec#*:}
    CKPT=${rest%%:*}
    PASS=${rest##*:}
    echo "--- $TAG pass $PASS ---"
    date -Is
    env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        nice -n 15 "$PY" -B -u eval_cohrot.py \
        --checkpoint "$CKPT" --label "$TAG" --pass-index "$PASS" \
        --limit 0 --out "$OUT" 2>&1 | tail -3
done
echo "=== chain complete ==="
date -Is
