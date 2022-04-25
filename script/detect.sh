# python tools/demo.py image -n yolox-s -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
# python tools/demo.py video -n yolox-s -c yolox_s.pth --path assets/MOT20-01-raw.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -f exps/pharma.py -n yolox-s -c latest_ckpt.pth --path inference/images --conf 0.25 --nms 0.5 --tsize 640 --save_result --device [cpu/gpu]
# python tools/demo.py video -f exps/pharma.py -n yolox-s -c last_epoch_ckpt.pth --path /mnt/nvme0n1/datasets/fisheye/20220411/videos/Sgpmc276-10.mp4 --conf 0.1 --nms 0.1 --tsize 640 --save_result --device [cpu/gpu]
