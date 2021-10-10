# custom training
python toonnx.py --scale=0.25 --ckpt="PPLCNet_x0_25_pretrained"
python toonnx.py --scale=0.35 --ckpt="PPLCNet_x0_35_pretrained"
python toonnx.py --scale=0.5 --ckpt="PPLCNet_x0_5_pretrained"
python toonnx.py --scale=0.75 --ckpt="PPLCNet_x0_75_pretrained"
python toonnx.py --scale=1.0 --ckpt="PPLCNet_x1_0_pretrained"
python toonnx.py --scale=1.5 --ckpt="PPLCNet_x1_5_pretrained"
python toonnx.py --scale=2.0 --ckpt="PPLCNet_x2_0_pretrained"
python toonnx.py --scale=2.5 --ckpt="PPLCNet_x2_5_pretrained"

## ssld training
python toonnx.py --scale=0.5 --ckpt="PPLCNet_x0_5_ssld_pretrained"
python toonnx.py --scale=1.0 --ckpt="PPLCNet_x1_0_ssld_pretrained"
python toonnx.py --scale=2.5 --ckpt="PPLCNet_x2_5_ssld_pretrained"