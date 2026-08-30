#!/usr/bin/env python3
"""Render a smooth, closed-loop latent-space walk from a trained
StyleGAN2-ADA checkpoint — the core "flowing data hallucination" effect
behind Refik Anadol's work.

Spherically interpolates through W-space between N random seeds and back to
the first, with ease-in/out timing per transition, writing one PNG per
frame. Assemble the frames into video with render_video.sh.

Must run ON THE POD from inside (or with PYTHONPATH pointing at) the
NVlabs/stylegan2-ada-pytorch checkout, since it imports that repo's
`dnnlib` and `legacy` modules:

    cd /workspace/stylegan2-ada-pytorch
    python /workspace/stylegang/scripts/generate_latent_walk.py \
      --network=path/to/network-snapshot-002000.pkl \
      --seeds=10 --frames-per-transition=90 \
      --outdir=/workspace/latent_walk_frames
"""
import argparse
import os

import numpy as np
import torch
import PIL.Image

import dnnlib
import legacy


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two same-shaped flat tensors."""
    a_n = a / a.norm()
    b_n = b / b.norm()
    omega = torch.acos(torch.clamp((a_n * b_n).sum(), -1.0, 1.0))
    if omega.abs() < 1e-6:
        return a + t * (b - a)
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b


def ease(t: float) -> float:
    """Smoothstep easing so each transition isn't constant-velocity."""
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, help="path or URL to a .pkl checkpoint")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", type=int, default=10, help="number of waypoint seeds in the loop")
    ap.add_argument("--frames-per-transition", type=int, default=90)
    ap.add_argument("--truncation-psi", type=float, default=0.7,
                     help="lower = more average/coherent, higher = more varied/extreme")
    ap.add_argument("--seed-start", type=int, default=0, help="base seed for reproducibility")
    ap.add_argument("--noise-mode", default="const", choices=["const", "random", "none"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading network from {args.network}...")
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    ws = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c = torch.zeros((1, G.c_dim), device=device) if G.c_dim != 0 else None
        w = G.mapping(z, c, truncation_psi=args.truncation_psi)
        ws.append(w)
    ws.append(ws[0])  # close the loop

    frame_idx = 0
    for i in range(len(ws) - 1):
        w_a, w_b = ws[i], ws[i + 1]
        shape = w_a.shape
        for f in range(args.frames_per_transition):
            t = ease(f / args.frames_per_transition)
            w_interp = slerp(w_a.flatten(), w_b.flatten(), t).reshape(shape)
            img = G.synthesis(w_interp, noise_mode=args.noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
                os.path.join(args.outdir, f"frame_{frame_idx:06d}.png")
            )
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  rendered {frame_idx} frames")

    print(f"done: {frame_idx} frames -> {args.outdir}")


if __name__ == "__main__":
    main()
