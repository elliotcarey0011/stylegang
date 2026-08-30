# stylegang

Refik Anadol–style generative art: fine-tune StyleGAN2-ADA on abstract WikiArt
imagery, then render smooth latent-space walk videos ("data hallucination"
loops) — trained and run remotely on a RunPod GPU pod.

Pipeline: **download dataset → prepare → train (transfer learning) → generate
latent walk frames → assemble video**.

## 0. Rent a RunPod pod

1. https://runpod.io → Deploy → Pod.
2. GPU: **RTX 4090** (24GB, ~$0.35–0.50/hr community cloud).
3. Template: `RunPod PyTorch 2.x` (comes with CUDA + PyTorch preinstalled).
4. Attach a **Network Volume** (e.g. 100GB) mounted at `/workspace` so your
   dataset and checkpoints survive if you stop/terminate the pod — training
   runs are cheap to pause and resume, don't pay to idle.
5. Connect via the web terminal or SSH.

## 1. Bootstrap the pod

```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch /workspace/stylegan2-ada-pytorch
cd /workspace/stylegan2-ada-pytorch
pip install click requests pyspng ninja imageio-ffmpeg==0.4.9 datasets pillow

# copy this repo's scripts in (scp/git-clone stylegang, or paste them in)
```

Get this `stylegang` repo onto the pod too (git clone it from wherever you
push it, or `scp -r` the `scripts/` folder up).

## 2. Download + curate the dataset

Pulls WikiArt from HuggingFace, filters to abstract/flowing styles (the
genres closest to Anadol's aesthetic — Abstract Expressionism, Color Field,
Cubism, etc.), center-crops to square, resizes to 512x512.

```bash
python scripts/download_dataset.py \
  --out /workspace/datasets/wikiart_raw \
  --resolution 512 \
  --styles abstract   # curated abstract-leaning styles, see --list-styles
```

Swap `--styles all` or pass explicit `--styles Impressionism,Fauvism` to
change the aesthetic. `--limit N` to cap image count for a quick test run.

## 3. Prepare the StyleGAN2-ADA dataset zip

```bash
cd /workspace/stylegan2-ada-pytorch
python dataset_tool.py \
  --source=/workspace/datasets/wikiart_raw \
  --dest=/workspace/datasets/wikiart512.zip \
  --width=512 --height=512
```

## 4. Train (transfer learning from FFHQ)

Edit `configs/train_config.env` if you want to change resolution/kimg, then:

```bash
bash scripts/train.sh
```

This resumes from NVIDIA's pretrained FFHQ-512 checkpoint with adaptive
discriminator augmentation (`--aug=ada`), which converges in hours instead of
days on a small/medium dataset. Snapshots (`fakesXXXXXX.png` grids +
`network-snapshot-XXXXXX.pkl`) land in `~/training-runs/`. You don't need
full convergence — for the Anadol "still hallucinating" look, an
under-trained snapshot (grainy, half-formed shapes) often looks better than a
fully converged one. Watch the fakes grids and pick a snapshot you like.

Stop the pod between training sessions to save money — the network volume
keeps your checkpoints.

## 5. Generate a latent-space walk

```bash
cd /workspace/stylegan2-ada-pytorch
python /workspace/stylegang/scripts/generate_latent_walk.py \
  --network=~/training-runs/00000-.../network-snapshot-002000.pkl \
  --seeds=10 \
  --frames-per-transition=90 \
  --outdir=/workspace/latent_walk_frames
```

Spherically interpolates through the W (style) space between N random seeds
in a smooth closed loop — this is the core "flowing data" Anadol effect.

## 6. Assemble the video

```bash
bash scripts/render_video.sh /workspace/latent_walk_frames /workspace/output.mp4
```

Encodes the PNG sequence to H.264 at 30fps and applies a light contrast/
saturation grade. Pull `output.mp4` off the pod (`scp` or the RunPod file
browser) when done.

## Notes

- Everything here runs on the remote GPU pod — this Mac has no CUDA, so none
  of these scripts are meant to run locally except `download_dataset.py`
  (which just needs network + `datasets`/`pillow`, no GPU).
- Cost-control: stop (don't just leave running) the pod when not actively
  training or rendering; the network volume persists your data.
