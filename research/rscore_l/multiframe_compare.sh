set -eu
runner=/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/rscore_l/launch.sh
task_root=/home/zhang/rscore-l-local
bash "$runner" train --variant lidar-multiframe > "$task_root/logs/train-lidar-multiframe.log" 2>&1
for split in val test; do
    bash "$runner" export --variant lidar-multiframe --split "$split" > "$task_root/logs/export-lidar-multiframe-$split.log" 2>&1
    bash "$runner" evaluate --variant lidar-multiframe --split "$split" > "$task_root/logs/evaluate-lidar-multiframe-$split.log" 2>&1
done
while [ ! -f "$task_root/evaluation/lidar/test/summary.json" ]; do
    if rg -q '"status": "FAILED"' "$task_root/state.json"; then
        exit 1
    fi
    sleep 30
done
bash "$runner" multiframe-report > "$task_root/logs/multiframe-report.log" 2>&1
