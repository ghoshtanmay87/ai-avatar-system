#!/usr/bin/env bash
# Runtime MuseTalk setup for the RunPod all-in-one container.
# Adapted from ../../scripts/setup_musetalk.sh: same steps (clone, patch
# preprocessing.py for CPU/GPU-agnostic face_alignment, install our
# persistent worker, download weights), but targets a path under the
# persistent /workspace volume and assumes all pip deps are already baked
# into the image (no venv). Idempotent — safe to re-run on every boot.
set -e

MODELS_ROOT="${1:?usage: setup_musetalk.sh <models_root> <musetalk_dir>}"
MUSETALK_DIR="${2:?usage: setup_musetalk.sh <models_root> <musetalk_dir>}"
BACKEND_DIR="/app/backend"
SENTINEL="$MODELS_ROOT/.musetalk_ready"
PYTHON=python3

if [ -f "$SENTINEL" ]; then
  echo "[musetalk] Sentinel found at $SENTINEL — models already set up, skipping."
  exit 0
fi

echo "=== MuseTalk V1.5 Setup (runtime, persistent volume) ==="
mkdir -p "$MODELS_ROOT"

# ── 1. Clone MuseTalk ────────────────────────────────────────────────────────
if [ -d "$MUSETALK_DIR/.git" ]; then
  echo "[1/4] MuseTalk already cloned — pulling latest..."
  git -C "$MUSETALK_DIR" pull --ff-only || true
else
  echo "[1/4] Cloning MuseTalk..."
  git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
fi

# ── 2. CPU/GPU-agnostic preprocessing.py (face_alignment instead of mmpose) ─
echo "[2/4] Writing face_alignment-based preprocessing.py..."
PREPROCESS="$MUSETALK_DIR/musetalk/utils/preprocessing.py"
cp "$PREPROCESS" "${PREPROCESS}.orig" 2>/dev/null || true
cat > "$PREPROCESS" << 'PYEOF'
import os
import json
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm
import face_alignment

_device = "cuda" if torch.cuda.is_available() else "cpu"
_fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device=_device,
)

coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def resize_landmark(landmark, w, h, new_w, new_h):
    landmark_norm = landmark / np.array([w, h])
    return landmark_norm * np.array([new_w, new_h])


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _detect_face_bbox(frame_bgr, bbox_shift=0):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds = _fa.get_landmarks(frame_rgb)
    if preds is None or len(preds) == 0:
        return None
    lm = preds[0]
    x1, y1 = int(lm[:, 0].min()), int(lm[:, 1].min())
    x2, y2 = int(lm[:, 0].max()), int(lm[:, 1].max())
    y1 = max(0, y1 + bbox_shift)
    if x2 <= x1 or y2 <= y1 or x1 < 0:
        return None
    return (x1, y1, x2, y2)


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    coords_list = []
    for frame in tqdm(frames):
        bbox = _detect_face_bbox(frame, bbox_shift=upperbondrange)
        coords_list.append(bbox if bbox is not None else coord_placeholder)
    return coords_list, frames


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    deltas = []
    for frame in tqdm(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = _fa.get_landmarks(frame_rgb)
        if preds is None or len(preds) == 0:
            continue
        lm = preds[0]
        deltas.append(int(lm[:, 1].max()) - int(lm[:, 1].mean()))
    if not deltas:
        return "No faces detected"
    avg = int(sum(deltas) / len(deltas))
    return f"Total frame: {len(frames)}  Adjust range: [-{avg}~{avg}]  current value: {upperbondrange}"
PYEOF

# ── 2b. Install our persistent worker driver ─────────────────────────────────
WORKER_SRC="$BACKEND_DIR/musetalk_worker.py"
WORKER_DST="$MUSETALK_DIR/scripts/musetalk_worker.py"
if [ -f "$WORKER_SRC" ]; then
  cp "$WORKER_SRC" "$WORKER_DST"
  echo "  musetalk_worker.py installed"
else
  echo "  WARNING: $WORKER_SRC not found — skipping worker install"
fi

# ── 3. Download model weights (~8.8GB, skips anything already present) ──────
echo "[3/4] Downloading MuseTalk model weights if missing..."
"$PYTHON" -m pip install -q gdown

"$PYTHON" - "$MUSETALK_DIR" << 'PYEOF'
from huggingface_hub import snapshot_download
import os, sys, urllib.request

musetalk_dir = sys.argv[1]
models_target = os.path.join(musetalk_dir, "models")
os.makedirs(models_target, exist_ok=True)

if not os.path.isfile(os.path.join(models_target, "musetalkV15", "unet.pth")):
    print("  Downloading TMElyralab/MuseTalk weights (~7 GB)...")
    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=models_target,
        ignore_patterns=["*.md", "*.txt", "*.gitattributes"],
    )
else:
    print("  MuseTalk weights already present — skipping.")

whisper_target = os.path.join(models_target, "whisper")
if not os.path.isdir(whisper_target) or not os.listdir(whisper_target):
    print("  Downloading openai/whisper-tiny (~150 MB)...")
    snapshot_download(
        repo_id="openai/whisper-tiny",
        local_dir=whisper_target,
        ignore_patterns=["*.md", "*.gitattributes", "flax_model*", "tf_model*", "rust_model*"],
    )
else:
    print("  Whisper-tiny already present — skipping.")

vae_target = os.path.join(models_target, "sd-vae")
if not os.path.isdir(vae_target) or not os.listdir(vae_target):
    print("  Downloading stabilityai/sd-vae-ft-mse (~335 MB)...")
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir=vae_target,
        ignore_patterns=["*.md", "*.gitattributes"],
    )
else:
    print("  SD-VAE already present — skipping.")

bisent_dir = os.path.join(models_target, "face-parse-bisent")
os.makedirs(bisent_dir, exist_ok=True)

resnet_path = os.path.join(bisent_dir, "resnet18-5c106cde.pth")
if not os.path.isfile(resnet_path):
    print("  Downloading ResNet18 backbone (~45 MB)...")
    urllib.request.urlretrieve(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        resnet_path,
    )
else:
    print("  resnet18-5c106cde.pth already present — skipping.")

bisenet_path = os.path.join(bisent_dir, "79999_iter.pth")
if not os.path.isfile(bisenet_path):
    print("  Downloading BiSeNet face parser via gdown (~53 MB)...")
    import gdown
    gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=bisenet_path, quiet=False)
else:
    print("  79999_iter.pth already present — skipping.")

print("  All model downloads complete.")
PYEOF

# ── 4. Verify ─────────────────────────────────────────────────────────────
echo "[4/4] Verifying installation..."
MISSING=0
for f in \
  "$MUSETALK_DIR/scripts/inference.py" \
  "$MUSETALK_DIR/scripts/musetalk_worker.py" \
  "$MUSETALK_DIR/models/musetalkV15/unet.pth" \
  "$MUSETALK_DIR/models/sd-vae/config.json" \
  "$MUSETALK_DIR/models/face-parse-bisent/resnet18-5c106cde.pth" \
  "$MUSETALK_DIR/models/face-parse-bisent/79999_iter.pth"
do
  if [ ! -f "$f" ]; then
    echo "  MISSING: $f"
    MISSING=1
  fi
done
[ -d "$MUSETALK_DIR/models/whisper" ] || { echo "  MISSING: models/whisper/"; MISSING=1; }

if [ "$MISSING" -eq 0 ]; then
  touch "$SENTINEL"
  echo "MuseTalk setup complete — sentinel written to $SENTINEL"
else
  echo "MuseTalk setup incomplete — see MISSING lines above" >&2
  exit 1
fi
