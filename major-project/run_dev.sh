#!/usr/bin/env bash
# Start the Green-Weight backend and dashboard together (local dev).
#
# The backend MUST run with cwd = backend/src/green_weight, not
# backend/src: every module uses bare same-directory imports
# (`from config import get_config`), so `-m green_weight.api` from
# backend/src fails. See CLAUDE.md, "Running the Project".
#
# The dashboard reaches the backend through Vite's /api proxy
# (vite.config.js), so both must be up for the UI to show "online".
#
# GREEN_WEIGHT_LOAD_MODELS=1 makes /infer attempt real GPU inference
# instead of the labelled mock responses (needs CUDA + HF_TOKEN).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend="$root/backend/src/green_weight"
frontend="$root/frontend"
backend_port="${BACKEND_PORT:-8000}"

[ -d "$frontend/node_modules" ] || (echo "Installing frontend dependencies (first run)..." && cd "$frontend" && npm install)

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Backend   -> http://localhost:$backend_port"
(cd "$backend" && python -m uvicorn api:app --reload --port "$backend_port") &

echo "Dashboard -> http://localhost:5173"
(cd "$frontend" && npm run dev) &

wait
