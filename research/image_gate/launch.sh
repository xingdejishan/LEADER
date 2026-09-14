set -eu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$1" = visual ]; then
    for libdir in /home/zhang/miniconda3/envs/bufferx/lib/python3.11/site-packages/nvidia/*/lib; do
        export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
    done
    exec /home/zhang/.venvs/rscore-l/bin/python "$script_dir/run.py" "$@"
elif [ "$1" = check ]; then
    exec /home/zhang/miniconda3/envs/egonn118/bin/python "$script_dir/check.py"
else
    exec /home/zhang/miniconda3/envs/egonn118/bin/python "$script_dir/run.py" "$@"
fi
