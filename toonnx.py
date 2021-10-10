
import argparse
import paddle
import onnx
import numpy as np
import onnxruntime
from lcnet import PPLCNetEngine
import subprocess


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0, help="")
    parser.add_argument("--ckpt", type=str, default="PPLCNet_x1_0_pretrained", help="")
    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    scale = args.scale
    ppckpt = args.ckpt
    oockpt = f"./onnx/{ppckpt}"

    model = PPLCNetEngine(scale=scale, pretrained=f"./paddle/{ppckpt}")
    model.eval()

    input_spec = paddle.static.InputSpec(shape=[1, 3, 224, 224], dtype='float32', name='data')
    paddle.onnx.export(model, oockpt, input_spec=[input_spec], opset_version=11)

    cmd = f"python -m onnxsim {oockpt}.onnx {oockpt}.onnx"
    subprocess.run(cmd, shell=True)

    onnx_file = f"{oockpt}.onnx"
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')

    inputs = paddle.randn((1, 3, 224, 224), "float32")
    ppout = model(inputs)

    ort_sess = onnxruntime.InferenceSession(onnx_file)
    print(ort_sess.get_inputs()[0].name)
    ort_inputs = {ort_sess.get_inputs()[0].name: inputs.numpy()}
    ooout = ort_sess.run(None, ort_inputs)

    diff = ppout.numpy() - ooout[0]
    print(f"Max Difference is : {np.abs(diff).max()}")


if __name__ == "__main__":
    main()