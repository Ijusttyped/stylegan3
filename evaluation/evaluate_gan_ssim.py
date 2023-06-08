import argparse
import json
import random

import numpy as np

import torch
import pytorch_ssim
import dnnlib
import legacy
from gen_images import make_transform


random.seed(42)


def calculate_ssim(gan_checkpoint, num_samples=1000):
    print('Loading networks from "%s"...' % gan_checkpoint)
    device = torch.device("cuda")
    with dnnlib.util.open_url(gan_checkpoint) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    sum_ssim = 0
    seeds = [random.randint(0, 100000) for _ in range(num_samples)]
    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(2, G.z_dim)).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, "input"):
            m = make_transform((0.0, 0.0), 0.0)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=1, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.float32)
        img1 = img[0].unsqueeze(dim=0).cpu()
        img2 = img[1].unsqueeze(dim=0).cpu()
        msssim = pytorch_ssim.msssim_3d(img1, img2)
        sum_ssim = sum_ssim + msssim
        print(sum_ssim / (seed_idx + 1.0))
    return sum_ssim / num_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_checkpoint", type=str)
    parser.add_argument("--out_path", type=str, default="output/gan_ssim.json")

    args = parser.parse_args()
    ssim = calculate_ssim(args.gan_checkpoint)
    print("-------------- SSIM --------------")
    print(ssim)
    with open(args.out_path, "w") as f:
        json.dump({"ssim": ssim.item()}, f)
