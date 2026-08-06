#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Single-container startup for AvatarAI on a RunPod Pod.
#
# RunPod Pods don't support nested Docker/Docker Compose, so everything the
# original docker-compose.yml split across 7 containers (postgres, redis,
# backend, celery-worker, frontend, nginx) runs here as supervised processes
# inside one image instead.
#
# Everything that must survive a pod stop/restart (Postgres data, MuseTalk
# model weights ~9GB, uploaded avatar photos/voice clones) lives under
# /workspace, which RunPod mounts from the pod's persistent Volume Disk.
# Everything else is on the ephemeral Container Disk and gets rebuilt from
# the image on every boot.
# ---------------------------------------------------------------------------
set -euo pipefail

log() { echo "[container-init] $*"; }

WORKSPACE=/workspace
PG_DATA="$WORKSPACE/postgres_data"
MUSETALK_ROOT="$WORKSPACE/musetalk_models"
MUSETALK_DIR="$MUSETALK_ROOT/MuseTalk"
UPLOADS_DIR="$WORKSPACE/uploads"
REDIS_DIR="$WORKSPACE/redis_data"
SECRETS_FILE="$WORKSPACE/.runtime_secrets"

mkdir -p "$PG_DATA" "$MUSETALK_ROOT" "$UPLOADS_DIR" "$REDIS_DIR" /var/log/supervisor
chown -R postgres:postgres "$PG_DATA"
chmod 700 "$PG_DATA"

# ── 1. Generate (or load) stable secrets so they survive restarts ──────────
if [ ! -f "$SECRETS_FILE" ]; then
  log "First boot — generating secrets into $SECRETS_FILE"
  {
    echo "export DATABASE_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    echo "export SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    echo "export JWT_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  } > "$SECRETS_FILE"
fi
# shellcheck disable=SC1090
source "$SECRETS_FILE"

export DATABASE_USER="${DATABASE_USER:-avatar_user}"
export DATABASE_NAME="${DATABASE_NAME:-avatar_db}"
export DATABASE_URL="postgresql://${DATABASE_USER}:${DATABASE_PASSWORD}@127.0.0.1:5432/${DATABASE_NAME}"
export REDIS_URL="redis://127.0.0.1:6379/0"
export CELERY_BROKER_URL="redis://127.0.0.1:6379/1"
export CELERY_RESULT_BACKEND="redis://127.0.0.1:6379/2"
export USE_LOCAL_STORAGE="${USE_LOCAL_STORAGE:-true}"
export LOCAL_STORAGE_PATH="$UPLOADS_DIR"
export MUSETALK_PATH="$MUSETALK_DIR"
export AVATAR_ENGINE="${AVATAR_ENGINE:-musetalk}"
export LLM_PROVIDER="${LLM_PROVIDER:-anthropic}"
export LLM_MODEL="${LLM_MODEL:-claude-sonnet-4-6}"
export ENVIRONMENT="${ENVIRONMENT:-production}"
export DEBUG="${DEBUG:-false}"
export CORS_ORIGINS="${CORS_ORIGINS:-*}"
# Frontend talks to the API/WS on the same origin (nginx proxies both) —
# leaving these unset makes the frontend build use relative paths, so it
# works no matter what URL RunPod assigns this pod.
export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-}"
export NEXT_PUBLIC_WS_URL="${NEXT_PUBLIC_WS_URL:-}"

log "DATABASE_URL=postgresql://${DATABASE_USER}:***@127.0.0.1:5432/${DATABASE_NAME}"

# ── 2. Initialise Postgres cluster on first boot ────────────────────────────
if [ ! -f "$PG_DATA/PG_VERSION" ]; then
  log "Initialising new Postgres cluster at $PG_DATA"
  su postgres -c "/usr/lib/postgresql/14/bin/initdb -D $PG_DATA --auth-local=trust --auth-host=trust" >/var/log/supervisor/initdb.log 2>&1

  cat >> "$PG_DATA/pg_hba.conf" <<-EOF
	local   all             all                                     trust
	host    all             all             127.0.0.1/32            trust
	host    all             all             ::1/128                 trust
	EOF

  log "Starting Postgres temporarily to create app role/database..."
  su postgres -c "/usr/lib/postgresql/14/bin/pg_ctl -D $PG_DATA -o '-c listen_addresses=127.0.0.1' -w start" >>/var/log/supervisor/initdb.log 2>&1
  su postgres -c "psql -v ON_ERROR_STOP=1 -c \"CREATE ROLE ${DATABASE_USER} LOGIN PASSWORD '${DATABASE_PASSWORD}';\""
  su postgres -c "psql -v ON_ERROR_STOP=1 -c \"CREATE DATABASE ${DATABASE_NAME} OWNER ${DATABASE_USER};\""
  su postgres -c "/usr/lib/postgresql/14/bin/pg_ctl -D $PG_DATA -w stop"
  log "Postgres cluster ready."
else
  log "Existing Postgres cluster found at $PG_DATA — reusing."
fi

# ── 3. MuseTalk models: clone repo + patch preprocessing + fetch weights ───
# Idempotent — safe to re-run on every boot; skips anything already present.
if [ "$AVATAR_ENGINE" = "musetalk" ]; then
  bash /app/deploy/runpod/setup_musetalk.sh "$MUSETALK_ROOT" "$MUSETALK_DIR" \
    || log "WARNING: MuseTalk setup failed — backend will fall back to non-musetalk mode."
fi

# ── 4. Redis persistence dir ─────────────────────────────────────────────────
chown -R redis:redis "$REDIS_DIR" 2>/dev/null || true

log "Handing off to supervisord..."
exec /usr/bin/supervisord -c /app/deploy/runpod/supervisord.conf
