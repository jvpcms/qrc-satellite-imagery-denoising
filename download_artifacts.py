"""Download all release artifacts — trained MLP models and cached reservoir
embeddings — from the GitHub Release into models/.

Per sigma in {0.1, 0.3, 0.5, 0.7}: the hybrid readout and classical baseline
models (~900 MB each, above GitHub's 100 MB per-file commit limit) and the
train/val/test reservoir-embedding arrays (~285 MB total). With these in
place, 01_pipeline.ipynb skips both reservoir simulation and MLP training —
no GPU needed. Total download ≈ 8.4 GB; already-present files are skipped.

Usage:
    python download_artifacts.py
"""
import urllib.request
from pathlib import Path

REPO = "jvpcms/qrc-satellite-imagery-denoising"
TAG = "1.0.0"
SIGMAS = [0.1, 0.3, 0.5, 0.7]
D_QRC = 18

MODEL_DIR = Path("models")


def fetch(asset: str):
    dest = MODEL_DIR / asset
    if dest.exists():
        print(f"{dest} already present — skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{REPO}/releases/download/{TAG}/{asset}"
    print(f"downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def main():
    for s in SIGMAS:
        fetch(f"sigma{s}_mlp_qrc_readout_d{D_QRC}.keras")
        fetch(f"sigma{s}_mlp_baseline_d{D_QRC}.keras")
        for split in ("train", "val", "test"):
            fetch(f"sigma{s}_embeddings_d{D_QRC}_{split}.npy")
    print("done")


if __name__ == "__main__":
    main()
