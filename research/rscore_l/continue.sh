set -eu
while kill -0 "${1:?node2vec process id required}" 2>/dev/null; do
    sleep 30
done
while kill -0 "${2:?geometry process id required}" 2>/dev/null; do
    sleep 30
done
bash /mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/rscore_l/launch.sh all > /home/zhang/rscore-l-local/pipeline.log 2>&1
