#!/usr/bin/env python3
"""Pull WikiArt from HuggingFace, filter by style, center-crop + resize to
square images, and dump to a flat folder for stylegan2-ada-pytorch's
dataset_tool.py to consume.

Runs anywhere with network access + `pip install datasets pillow` — no GPU
needed, so it's fine to run locally too if you'd rather stage the dataset
before uploading it to the pod.
"""
import argparse
from pathlib import Path

from PIL import Image

# Curated subset of WikiArt "style" classes closest to Refik Anadol's
# flowing, abstract, non-figurative aesthetic.
ABSTRACT_STYLES = [
    "Abstract_Expressionism",
    "Action_painting",
    "Color_Field_Painting",
    "Cubism",
    "Analytical_Cubism",
    "Synthetic_Cubism",
    "Fauvism",
    "Minimalism",
    "Pointillism",
]

ALL_STYLES = [
    "Abstract_Expressionism", "Action_painting", "Analytical_Cubism",
    "Art_Nouveau", "Baroque", "Color_Field_Painting", "Contemporary_Realism",
    "Cubism", "Early_Renaissance", "Expressionism", "Fauvism",
    "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance",
    "Minimalism", "Naive_Art_Primitivism", "New_Realism",
    "Northern_Renaissance", "Pointillism", "Pop_Art", "Post_Impressionism",
    "Realism", "Rococo", "Romanticism", "Symbolism", "Synthetic_Cubism",
    "Ukiyo_e",
]


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output image folder")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument(
        "--styles",
        default="abstract",
        help="'abstract' (curated Anadol-leaning subset), 'all', or a "
        "comma-separated list of WikiArt style names (see --list-styles)",
    )
    ap.add_argument("--limit", type=int, default=None,
                     help="cap number of saved images (default: all matching)")
    ap.add_argument("--list-styles", action="store_true")
    args = ap.parse_args()

    if args.list_styles:
        print("\n".join(ALL_STYLES))
        return

    if args.styles == "abstract":
        wanted = set(ABSTRACT_STYLES)
    elif args.styles == "all":
        wanted = None
    else:
        wanted = set(s.strip() for s in args.styles.split(","))
        unknown = wanted - set(ALL_STYLES)
        if unknown:
            raise SystemExit(f"unknown style(s): {unknown}\nvalid: {ALL_STYLES}")

    from datasets import load_dataset

    print(f"loading huggan/wikiart (streaming)... filter={wanted or 'all'}")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    style_names = ds.features["style"].names

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, row in enumerate(ds):
        style_name = style_names[row["style"]]
        if wanted is not None and style_name not in wanted:
            continue

        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = center_crop_square(img)
        img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
        img.save(out_dir / f"{saved:06d}.png")
        saved += 1

        if saved % 200 == 0:
            print(f"  saved {saved} images (scanned {i + 1})")
        if args.limit and saved >= args.limit:
            break

    print(f"done: {saved} images -> {out_dir}")


if __name__ == "__main__":
    main()
