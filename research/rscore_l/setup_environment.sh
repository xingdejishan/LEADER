set -eu
ROOT=/mnt/c/Users/zhang/Documents/ChatGPT/LEADER
PY=/home/zhang/.venvs/rscore-l/bin/python
mkdir -p /home/zhang/.cache/torch/hub/checkpoints /home/zhang/.cache/torch/hub/netvlad
ln -sf "$ROOT/rscore-assets/dedode_detector_L.pth" /home/zhang/.cache/torch/hub/checkpoints/dedode_detector_L.pth
ln -sf "$ROOT/rscore-assets/dedode_descriptor_B.pth" /home/zhang/.cache/torch/hub/checkpoints/dedode_descriptor_B.pth
ln -sf "$ROOT/rscore-assets/Pitts30K_struct.mat" /home/zhang/.cache/torch/hub/netvlad/VGG16-NetVLAD-Pitts30K.mat
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
bash "$ROOT/LEADER/research/rscore_l/launch.sh" prepare
