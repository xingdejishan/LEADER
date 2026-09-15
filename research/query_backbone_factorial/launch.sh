set -eu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$1" = all ]; then
    shift
    for stage in check prepare extract compress train evaluate report; do
        bash "$script_dir/launch.sh" "$stage" "$@"
    done
    exit
fi
if [ "$1" = extract ]; then
    for libdir in /home/zhang/miniconda3/envs/bufferx/lib/python3.11/site-packages/nvidia/*/lib; do
        export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
    done
    exec /home/zhang/.venvs/rscore-l/bin/python "$script_dir/run.py" "$@"
fi
exec /home/zhang/miniconda3/envs/egonn118/bin/python "$script_dir/run.py" "$@"
