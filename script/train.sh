<<<<<<< HEAD
python tools/train.py -f exps/pharma.py -d 1 -b 4 --fp16 -o -c last_epoch_ckpt.pth
=======
python tools/train.py -f exps/rotated_ds.py -d 1 -b 12 --fp16 -o -c yolox_s.pth
>>>>>>> 8429c8b756e7f8fc52c095c87f3cbd967aa69056
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f exps/p50.py -d 1 -b 1 --fp16 -o -c yolox_s.pth