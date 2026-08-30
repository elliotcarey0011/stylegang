# stylegang

Refik Anadol–style generative art: fine-tune StyleGAN2-ADA on abstract WikiArt
imagery, then render smooth latent-space walk videos ("data hallucination"
loops) — trained and run remotely on a RunPod GPU pod.

Pipeline: **download dataset → prepare → train (transfer learning) → generate
latent walk frames → assemble video**.

## 0. Rent a RunPod pod

1. https://runpod.io → Deploy → Pod.
2. GPU: **RTX 4090** (24GB, ~$0.35–0.50/hr community cloud). Ada Lovelace
   GPUs need CUDA ≥11.8 to even be recognized, so any current RunPod CUDA
   12.x template satisfies that automatically.
3. Template: the official **PyTorch template on a CUDA 12.4 "devel" image**
   (shown in the gallery as "PyTorch 2.4 + CUDA 12.4", tag looks like
   `runpod/pytorch:...-cu1241-torch240-devel-ubuntu2204`). Pick **devel**,
   not runtime — it ships `nvcc`, which stylegan2-ada-pytorch needs to
   JIT-compile its custom `bias_act`/`upfirdn2d` CUDA kernels for full
   speed. Don't chase the NVlabs README's literal ask of Python 3.7 /
   PyTorch 1.7.1 / CUDA 11.0 — that's a 2020-era pin RunPod no longer even
   offers, and it predates Ada Lovelace support. If the custom kernels ever
   fail to compile against a newer torch, the repo silently falls back to a
   slower pure-PyTorch implementation rather than breaking, so a version
   mismatch costs speed, not correctness. If you want the compiled-kernel
   speed and hit a build error, this is a well-reported working downgrade
   inside the container: `pip install torch==2.0.1 torchvision==0.15.2
   --index-url https://download.pytorch.org/whl/cu118`.
4. Attach a **Network Volume** (e.g. 100GB) mounted at `/workspace` so your
   dataset and checkpoints survive if you stop/terminate the pod — training
   runs are cheap to pause and resume, don't pay to idle.
5. Connect via the web terminal or SSH.

## 1. Bootstrap the pod

```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch /workspace/stylegan2-ada-pytorch
cd /workspace/stylegan2-ada-pytorch
pip install click requests pyspng ninja imageio-ffmpeg==0.4.9 datasets pillow

# get this repo's scripts onto the pod too
git clone https://github.com/elliotcarey0011/stylegang /workspace/stylegang
```

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
