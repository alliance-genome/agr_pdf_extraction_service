#!/bin/bash

# Deployment script for agr_pdf_extraction_service on FlySQL servers
# Adapted from ai_curation_prototype/backend/docling-service/deploy.sh
#
# Usage: ./deploy.sh
#
# Prerequisites:
#   - Docker and Docker Compose installed
#   - .env file configured (copy from .env.example)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_NAME="pdfx"
GPU_MODE="${GPU_MODE:-auto}" # auto|on|off
PDFX_DEPLOY_BUILD_MODE="${PDFX_DEPLOY_BUILD_MODE:-auto}" # auto|rebuild|never
PDFX_DEPLOY_PULL_IMAGES="${PDFX_DEPLOY_PULL_IMAGES:-auto}" # auto|always|never

detect_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

select_compose_file() {
    case "${GPU_MODE}" in
        on)
            if ! detect_gpu; then
                echo "ERROR: GPU_MODE=on but no NVIDIA GPU detected (or nvidia-smi unavailable)." >&2
                return 1
            fi
            echo "docker-compose.gpu.yml"
            ;;
        off)
            echo "docker-compose.yml"
            ;;
        auto)
            if detect_gpu; then
                echo "docker-compose.gpu.yml"
            else
                echo "docker-compose.yml"
            fi
            ;;
        *)
            echo "ERROR: Invalid GPU_MODE='${GPU_MODE}'. Use auto|on|off." >&2
            return 1
            ;;
    esac
}

if ! COMPOSE_FILE="$(select_compose_file)"; then
    exit 1
fi
COMPOSE_ARGS=(-f "$COMPOSE_FILE" -p "$PROJECT_NAME")
BUILD_ARGS=()

case "${PDFX_DEPLOY_BUILD_MODE}" in
    auto)
        # Let Docker Compose build only when the local image is missing.
        ;;
    rebuild)
        BUILD_ARGS=(--build)
        ;;
    never)
        BUILD_ARGS=(--no-build)
        ;;
    *)
        echo "ERROR: Invalid PDFX_DEPLOY_BUILD_MODE='${PDFX_DEPLOY_BUILD_MODE}'. Use auto|rebuild|never." >&2
        exit 1
        ;;
esac

case "${PDFX_DEPLOY_PULL_IMAGES}" in
    auto|always|never)
        ;;
    *)
        echo "ERROR: Invalid PDFX_DEPLOY_PULL_IMAGES='${PDFX_DEPLOY_PULL_IMAGES}'. Use auto|always|never." >&2
        exit 1
        ;;
esac

SHOULD_PULL_PREBUILT_GPU_IMAGE=false
if [ "${PDFX_DEPLOY_PULL_IMAGES}" = "always" ]; then
    SHOULD_PULL_PREBUILT_GPU_IMAGE=true
elif [ "${PDFX_DEPLOY_PULL_IMAGES}" = "auto" ] && [ "${PDFX_DEPLOY_BUILD_MODE}" = "never" ]; then
    SHOULD_PULL_PREBUILT_GPU_IMAGE=true
fi

if [ "$COMPOSE_FILE" = "docker-compose.gpu.yml" ] && \
   [ -n "${PDFX_GPU_IMAGE:-}" ] && \
   [ "${PDFX_DEPLOY_BUILD_MODE}" = "never" ]; then
    COMPOSE_ARGS=(-f "$COMPOSE_FILE" -f "docker-compose.gpu.prebuilt.yml" -p "$PROJECT_NAME")
fi

if [ "$COMPOSE_FILE" = "docker-compose.gpu.yml" ]; then
    GROBID_HEALTH_PORT="8070"
    echo "GPU deployment mode selected (${GPU_MODE}); using ${COMPOSE_FILE}"
else
    GROBID_HEALTH_PORT="${PDFX_GROBID_PORT:-8075}"
    echo "CPU deployment mode selected (${GPU_MODE}); using ${COMPOSE_FILE}"
fi

echo "=== PDF Extraction Service Deployment ==="

wait_for_postgres() {
    local attempts=60
    local delay=5

    echo "Waiting for Postgres to accept connections..."
    for attempt in $(seq 1 "$attempts"); do
        if docker exec pdfx-postgres pg_isready -U pdfx >/dev/null 2>&1; then
            echo "  Postgres is ready."
            return 0
        fi

        echo "  Postgres not ready yet (${attempt}/${attempts}); retrying in ${delay}s..."
        sleep "$delay"
    done

    echo "ERROR: Postgres did not become ready after $((attempts * delay)) seconds."
    return 1
}

probe_gpu_image() {
    local image="${PDFX_GPU_IMAGE:-pdfx-gpu}"

    docker run --rm --gpus all --entrypoint python3.11 "$image" -c '
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("CUDA is not available inside the GPU image")

torch.cuda.mem_get_info(0)
print(torch.cuda.get_device_name(0))
'
}

wait_for_nvidia_container_runtime() {
    if [ "$COMPOSE_FILE" != "docker-compose.gpu.yml" ]; then
        return 0
    fi

    local image="${PDFX_GPU_IMAGE:-pdfx-gpu}"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        if [ "${PDFX_DEPLOY_BUILD_MODE}" = "auto" ]; then
            echo "Skipping pre-start NVIDIA image probe; ${image} will be built by Docker Compose if needed."
            return 0
        fi
    fi

    local attempts="${PDFX_NVIDIA_READY_ATTEMPTS:-36}"
    local delay="${PDFX_NVIDIA_READY_DELAY_SECONDS:-5}"

    echo "Waiting for NVIDIA container runtime..."
    for attempt in $(seq 1 "$attempts"); do
        if detect_gpu && probe_gpu_image >/tmp/pdfx-gpu-probe.out 2>/tmp/pdfx-gpu-probe.err; then
            echo "  NVIDIA runtime is ready ($(cat /tmp/pdfx-gpu-probe.out))."
            return 0
        fi

        echo "  NVIDIA runtime not ready (${attempt}/${attempts}); retrying in ${delay}s..."
        if [ -s /tmp/pdfx-gpu-probe.err ]; then
            sed 's/^/    /' /tmp/pdfx-gpu-probe.err | tail -n 8
        fi
        sleep "$delay"
    done

    echo "ERROR: NVIDIA container runtime did not become ready after $((attempts * delay)) seconds."
    return 1
}

probe_worker_cuda() {
    docker exec pdfx-worker python3.11 -c '
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("CUDA is not available inside pdfx-worker")

torch.cuda.mem_get_info(0)
print(torch.cuda.get_device_name(0))
'
}

# Step 1: Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: Docker Compose is not installed"
    exit 1
fi

if [ ! -f ".env" ]; then
    if [ ! -f "$REPO_ROOT/.env" ]; then
        echo "ERROR: .env file not found. Copy from .env.example and configure."
        exit 1
    fi
    cd "$SCRIPT_DIR"
elif [ "$PWD" != "$SCRIPT_DIR" ]; then
    cd "$SCRIPT_DIR"
fi

if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "ERROR: .env file not found at $REPO_ROOT/.env. Copy from .env.example and configure."
    exit 1
fi

# Step 2: Create persistent directories
echo "Creating persistent directories..."
mkdir -p "$REPO_ROOT/data/cache"
mkdir -p "$REPO_ROOT/data/uploads"
mkdir -p "$REPO_ROOT/data/models"
mkdir -p "$REPO_ROOT/data/model_cache"
mkdir -p "$REPO_ROOT/data/rapidocr_models"
mkdir -p "$REPO_ROOT/logs"

# Step 3: Stop existing services
echo "Stopping existing services..."
docker compose "${COMPOSE_ARGS[@]}" down 2>/dev/null || true

# Step 4: Prepare application image and start dependency services
if [ "$COMPOSE_FILE" = "docker-compose.gpu.yml" ] && \
   [ -n "${PDFX_GPU_IMAGE:-}" ] && \
   [ "${SHOULD_PULL_PREBUILT_GPU_IMAGE}" = "true" ]; then
    echo "Pulling prebuilt GPU application image: ${PDFX_GPU_IMAGE}"
    docker compose "${COMPOSE_ARGS[@]}" pull app worker
fi

if [ "${PDFX_DEPLOY_BUILD_MODE}" = "rebuild" ]; then
    echo "Rebuilding application images before migrations..."
    docker compose "${COMPOSE_ARGS[@]}" build app worker
fi

wait_for_nvidia_container_runtime

echo "Starting dependency services..."
docker compose "${COMPOSE_ARGS[@]}" up -d postgres redis grobid

# Step 5: Wait for dependency services to start
echo "Waiting for services to start..."
wait_for_postgres

# Step 6: Run database migrations
echo "Running database migrations..."
docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps app alembic upgrade head
echo "  Database migrations applied."

# Step 7: Start full stack
echo "Starting application services (build mode: ${PDFX_DEPLOY_BUILD_MODE})..."
docker compose "${COMPOSE_ARGS[@]}" up -d "${BUILD_ARGS[@]}"

# Step 8: Health checks
echo ""
echo "=== Health Checks ==="

# Check GROBID
echo -n "  GROBID: "
if curl -sf "http://localhost:${GROBID_HEALTH_PORT}/api/isalive" > /dev/null 2>&1; then
    echo "HEALTHY"
else
    echo "NOT READY (may still be loading models, check: docker compose ${COMPOSE_ARGS[*]} logs grobid)"
fi

# Check Redis
echo -n "  Redis: "
if docker exec pdfx-redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "HEALTHY"
else
    echo "NOT READY"
fi

# Check Postgres
echo -n "  Postgres: "
if docker exec pdfx-postgres pg_isready -U pdfx 2>/dev/null | grep -q "accepting connections"; then
    echo "HEALTHY"
else
    echo "NOT READY"
fi

# Check Flask app
echo -n "  Flask app: "
if curl -sf http://localhost:5000/api/v1/health > /dev/null 2>&1; then
    echo "HEALTHY"
else
    echo "NOT READY (may still be loading models)"
fi

if [ "$COMPOSE_FILE" = "docker-compose.gpu.yml" ]; then
    echo -n "  GPU worker CUDA: "
    if probe_worker_cuda >/tmp/pdfx-worker-cuda-probe.out 2>/tmp/pdfx-worker-cuda-probe.err; then
        echo "HEALTHY ($(cat /tmp/pdfx-worker-cuda-probe.out))"
    else
        echo "NOT READY"
        if [ -s /tmp/pdfx-worker-cuda-probe.err ]; then
            sed 's/^/    /' /tmp/pdfx-worker-cuda-probe.err | tail -n 8
        fi
        exit 1
    fi
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Service endpoints:"
echo "  Web UI:    http://localhost:5000"
echo "  REST API:  http://localhost:5000/api/v1/"
echo "  Health:    http://localhost:5000/api/v1/health"
echo "  GROBID:    http://localhost:${GROBID_HEALTH_PORT}"
echo ""
echo "Management:"
echo "  Logs:      docker compose ${COMPOSE_ARGS[*]} logs -f"
echo "  Status:    ./manage.sh status"
echo "  Stop:      docker compose ${COMPOSE_ARGS[*]} down"
