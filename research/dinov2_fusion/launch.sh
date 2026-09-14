set -eu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
for libdir in /home/zhang/miniconda3/envs/bufferx/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
done
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$1" = train ] || [ "$1" = evaluate ]; then
    unset LD_LIBRARY_PATH
    exec /home/zhang/miniconda3/envs/egonn118/bin/python "$script_dir/run.py" "$@"
fi
if [ "$1" = all ]; then
    shift
    bash "$script_dir/launch.sh" prepare "$@"
    bash "$script_dir/launch.sh" extract "$@"
    bash "$script_dir/launch.sh" train "$@"
    exec bash "$script_dir/launch.sh" evaluate "$@"
fi
exec /home/zhang/.venvs/rscore-l/bin/python "$script_dir/run.py" "$@"
