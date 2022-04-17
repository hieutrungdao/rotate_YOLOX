python tools/train.py -f exps/rotated_ds.py -d 1 -b 12 --fp16 -o -c yolox_s.pth
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/p50.py -d 1 -b 1 --fp16 -o -c yolox_s.pth