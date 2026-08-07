# ---------------------------------------------------------------------------
# All-in-one AvatarAI image for a single RunPod Pod.
#
# RunPod Pods are themselves containers and don't support nested Docker /
# Docker Compose, so this bundles everything docker-compose.yml normally
# splits across 7 containers (postgres, redis, backend, celery-worker,
# frontend, flower, nginx) into one image, run as supervised processes.
#
# Build (from repo root):
#   docker build -f deploy/runpod/Dockerfile -t <you>/avatarai-runpod:latest .
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# ── OS packages: Python, ffmpeg, Postgres, Redis, nginx, supervisor, Node ───
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      ffmpeg libsm6 libxext6 libxrender-dev libgomp1 \
      git wget curl gnupg ca-certificates lsb-release build-essential \
      postgresql-14 postgresql-contrib \
      redis-server \
      nginx \
      supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && ln -sf python3.10 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    # We manage our own Postgres cluster on the persistent volume at runtime
    # (see container-init.sh) — drop the cluster the apt postinst created.
    && pg_dropcluster --stop 14 main || true \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Backend: Python deps ────────────────────────────────────────────────────
# chatterbox-tts>=0.1.4 (all published versions) hard-pins torch==2.6.0 /
# torchaudio==2.6.0, which conflicts with this repo's torch==2.2.0 pin (and,
# transitively, with MuseTalk's own diffusers==0.32.2 requirement — newer
# chatterbox-tts also wants diffusers==0.29.0, an unresolvable double
# conflict). Rather than upgrade the whole ML stack and risk breaking
# MuseTalk, we drop chatterbox-tts here: the backend's TTS service already
# has a built-in fallback chain (see app/services/tts.py) that catches
# exactly this "chatterbox unavailable" case and uses Edge TTS's free
# neural voices instead. You lose voice *cloning*; everything else —
# including the avatar actually speaking — keeps working.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN grep -v '^chatterbox-tts' /app/backend/requirements.txt > /tmp/requirements.filtered.txt \
    && pip install --no-cache-dir -r /tmp/requirements.filtered.txt \
    && pip install --no-cache-dir \
         torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 \
         --index-url https://download.pytorch.org/whl/cu118 \
    # extra deps used by MuseTalk's preprocessing/inference pipeline that
    # aren't already pulled in by backend/requirements.txt
    && pip install --no-cache-dir \
         einops==0.8.1 omegaconf "moviepy<2" mediapipe safetensors timm gdown huggingface_hub \
         numpy==1.26.3

COPY backend/ /app/backend/
RUN chmod +x /app/backend/entrypoint.sh \
    && mkdir -p /app/backend/voice_profiles /tmp/avatars /tmp/videos /tmp/audio

# ── Frontend: build with relative API/WS paths (nginx proxies same-origin,
#    so we don't need to know the pod's URL at build time) ─────────────────
COPY frontend/package.json frontend/package-lock.json* /app/frontend/
RUN cd /app/frontend && npm install
COPY frontend/ /app/frontend/
ENV NEXT_PUBLIC_API_URL="" \
    NEXT_PUBLIC_WS_URL="" \
    NEXT_PUBLIC_SITE_URL=""
RUN cd /app/frontend && npm run build

# ── Deployment glue: nginx conf, supervisord conf, boot script ─────────────
COPY deploy/runpod/ /app/deploy/runpod/
RUN chmod +x /app/deploy/runpod/container-init.sh /app/deploy/runpod/setup_musetalk.sh

EXPOSE 80

CMD ["/app/deploy/runpod/container-init.sh"]
