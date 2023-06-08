import argparse
import json
import random
from pathlib import Path

import torch
import dnnlib
import legacy
from metrics.frechet_inception_distance import compute_fid
from metrics.metric_utils import MetricOptions


def calculate_fid(gan_checkpoint, num_samples=2000):
    training_options_path = Path(gan_checkpoint).parent / "training_options.json"
    print(f"Loading training options from {training_options_path}")
    with open(training_options_path.as_posix(), "r") as f:
        training_options = json.load(f)
    dataset_kwargs = training_options["training_set_kwargs"]

    print('Loading networks from "%s"...' % gan_checkpoint)
    device = torch.device("cuda")
    with dnnlib.util.open_url(gan_checkpoint) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    print(f"Calculating FID for {num_samples} generated images...")
    opts = MetricOptions(G=G, G_kwargs={}, dataset_kwargs=dataset_kwargs)
    fid_val = compute_fid(
        opts=opts,
        max_real=None,
        num_gen=num_samples,
    )
    return fid_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_checkpoint", type=str)
    parser.add_argument("--out_path", type=str, default="output/gan_fid.json")

    args = parser.parse_args()
    fid = calculate_fid(args.gan_checkpoint)
    print("-------------- FID --------------")
    print(fid)
    with open(args.out_path, "w") as f:
        json.dump({"FID": fid}, f)
