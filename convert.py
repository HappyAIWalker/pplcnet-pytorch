
import argparse
import numpy as np
import torch
import paddle
from collections import OrderedDict
from torchlcnet import TorchLCNet
from lcnet import PPLCNetEngine


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
    ttckpt = f"./torch/{ppckpt}.pth.tar"
    print(f"Converting {ppckpt}")

    torchnet = TorchLCNet(scale=scale)
    ppnet = PPLCNetEngine(scale=scale, pretrained=f"./paddle/{ppckpt}")

    checkpoint = ppnet.state_dict()

    state_dict = OrderedDict()
    for key, val in torchnet.state_dict().items():
        # print(key)
        if "batches" in key:
            continue

        if "_mean" in key:
            ppkey = key.replace("running_mean", "_mean")
        elif "_var" in key:
            ppkey = key.replace("running_var", "_variance")
        else:
            ppkey = key

        state_dict[key] = torch.from_numpy(checkpoint[ppkey].numpy())
        # print(f"Key: {key} vs {ppkey}, size: {val.shape} vs {checkpoint[ppkey].shape}")

    state_dict["fc.weight"] = torch.transpose(state_dict["fc.weight"], 1, 0)

    torchnet.load_state_dict(state_dict)

    ppnet.eval()
    torchnet.eval()

    input = paddle.rand((1, 3, 224, 224), "float32")

    ppout = ppnet(input)
    ttout = torchnet(torch.from_numpy(input.numpy()))

    diff = ppout.numpy() - ttout.detach().numpy()
    print(np.abs(diff).max())

    # save checkpoint
    torch.save(torchnet.state_dict(), ttckpt)


if __name__ == "__main__":
    main()
