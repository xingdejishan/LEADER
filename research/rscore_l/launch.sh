set -eu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
for libdir in /home/zhang/miniconda3/envs/bufferx/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
done
exec /home/zhang/.venvs/rscore-l/bin/python /mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/rscore_l/run.py "$@"
