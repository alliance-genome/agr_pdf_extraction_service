#!/bin/bash

# Management script for agr_pdf_extraction_service
# Adapted from ai_curation_prototype/backend/docling-service/manage.sh

PROJECT_NAME="pdfx"
GPU_MODE="${GPU_MODE:-auto}" # auto|on|off

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

if [ "$COMPOSE_FILE" = "docker-compose.gpu.yml" ]; then
    GROBID_HEALTH_PORT="8070"
else
    GROBID_HEALTH_PORT="${PDFX_GROBID_PORT:-8075}"
fi

redis_container="pdfx-redis"
postgres_container="pdfx-postgres"
app_container="pdfx-app"
worker_container="pdfx-worker"

case "$1" in
    start)
        echo "Starting PDF extraction service..."
        docker compose "${COMPOSE_ARGS[@]}" up -d
        ;;

    stop)
        echo "Stopping PDF extraction service..."
        docker compose "${COMPOSE_ARGS[@]}" down
        ;;

    restart)
        echo "Restarting PDF extraction service..."
        docker compose "${COMPOSE_ARGS[@]}" down
        docker compose "${COMPOSE_ARGS[@]}" up -d
        ;;

    status)
        echo "=== Container Status ==="
        docker compose "${COMPOSE_ARGS[@]}" ps
        echo ""
        echo "=== Health Checks ==="
        echo -n "  GROBID:  "
        curl -s "http://localhost:${GROBID_HEALTH_PORT}/api/isalive" > /dev/null 2>&1 && echo "HEALTHY" || echo "DOWN"
        echo -n "  Redis:   "
        docker exec "${redis_container}" redis-cli ping 2>/dev/null | grep -q PONG && echo "HEALTHY" || echo "DOWN"
        echo -n "  Postgres:"
        docker exec "${postgres_container}" pg_isready -U pdfx 2>/dev/null | grep -q "accepting connections" && echo " HEALTHY" || echo " DOWN"
        echo -n "  App:     "
        curl -s http://localhost:5000/api/v1/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "DOWN"
        ;;

    logs)
        docker compose "${COMPOSE_ARGS[@]}" logs -f ${2:-}
        ;;

    logs-tail)
        docker compose "${COMPOSE_ARGS[@]}" logs --tail 100 ${2:-}
        ;;

    shell)
        echo "Opening shell in app container..."
        docker exec -it "${app_container}" /bin/bash
        ;;

    rebuild)
        echo "Rebuilding and redeploying..."
        docker compose "${COMPOSE_ARGS[@]}" down
        docker compose "${COMPOSE_ARGS[@]}" up -d --build
        ;;

    test)
        if [ -z "$2" ]; then
            echo "Usage: $0 test <pdf_file>"
            exit 1
        fi
        if [ ! -f "$2" ]; then
            echo "File not found: $2"
            exit 1
        fi
        echo "Submitting $2 for extraction..."
        curl -X POST \
            -F "file=@$2" \
            -F "methods=grobid" \
            -F "methods=docling" \
            -F "methods=marker" \
            -F "merge=on" \
            http://localhost:5000/process | python3 -m json.tool
        ;;

    worker-status)
        echo "=== Celery Worker Status ==="
        docker exec "${worker_container}" celery -A celery_app inspect active 2>/dev/null || echo "No workers running"
        ;;

    cleanup)
        echo "Cleaning up Docker resources..."
        docker compose "${COMPOSE_ARGS[@]}" down -v
        echo "Cleanup complete (volumes removed)"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs|logs-tail|shell|rebuild|test|worker-status|cleanup}"
        echo "Env:   GPU_MODE=auto|on|off (default: auto)"
        echo ""
        echo "Commands:"
        echo "  start          - Start all services"
        echo "  stop           - Stop all services"
        echo "  restart        - Restart all services"
        echo "  status         - Check service status and health"
        echo "  logs [svc]     - Follow logs (optional: specify service)"
        echo "  logs-tail [svc]- Show last 100 log lines"
        echo "  shell          - Open shell in app container"
        echo "  rebuild        - Rebuild images and redeploy"
        echo "  test <pdf>     - Test with a PDF file"
        echo "  worker-status  - Show Celery worker status"
        echo "  cleanup        - Remove containers AND volumes"
        exit 1
        ;;
esac
