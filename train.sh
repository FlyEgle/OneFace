cd /data/remote/github_code/OneFace;
python -W ignore train_net.py \
--dist-url 'tcp://127.0.0.1:9999' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \
