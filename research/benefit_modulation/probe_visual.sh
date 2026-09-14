set -eu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
for libdir in /home/zhang/miniconda3/envs/bufferx/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
done
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /home/zhang/.venvs/rscore-l/bin/python "$script_dir/probe_data.py" visual
