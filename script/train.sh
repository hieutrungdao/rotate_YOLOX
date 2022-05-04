# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/nano.py -d 1 -b 6 --fp16 -o -c yolox_nano.pth 
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/p50.py -d 1 -b 1 --fp16 -o -c yolox_s.pth
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/nano.py -d 1 -b 6 --fp16 -o
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/nano.py -d 1 -b 8 --fp16 -o --resume
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/nano.py -d 1 -b 6 --fp16 -o -c pretrain.pth
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/p50.py -d 1 -b 1 --fp16 -o -c epoch_196_ckpt.pth
CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/nano.py -d 1 -b 8 --fp16 -o -c epoch_90_ckpt.pth