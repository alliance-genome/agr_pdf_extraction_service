#!/bin/bash

# Management script for agr_pdf_extraction_service
# Adapted from ai_curation_prototype/backend/docling-service/manage.sh

PROJECT_NAME="pdfx"

case "$1" in
    start)
        echo "Starting PDF extraction service..."
        docker compose -p $PROJECT_NAME up -d
        ;;

    stop)
        echo "Stopping PDF extraction service..."
        docker compose -p $PROJECT_NAME down
        ;;

    restart)
        echo "Restarting PDF extraction service..."
        docker compose -p $PROJECT_NAME down
        docker compose -p $PROJECT_NAME up -d
        ;;

    status)
        echo "=== Container Status ==="
        docker compose -p $PROJECT_NAME ps
        echo ""
        echo "=== Health Checks ==="
        echo -n "  GROBID:  "
        curl -s http://localhost:8070/api/isalive > /dev/null 2>&1 && echo "HEALTHY" || echo "DOWN"
        echo -n "  Redis:   "
        docker exec ${PROJECT_NAME}-redis redis-cli ping 2>/dev/null | grep -q PONG && echo "HEALTHY" || echo "DOWN"
        echo -n "  Postgres:"
        docker exec pdfx-postgres pg_isready -U pdfx 2>/dev/null | grep -q "accepting connections" && echo " HEALTHY" || echo " DOWN"
        echo -n "  App:     "
        curl -s http://localhost:5000/api/v1/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "DOWN"
        ;;

    logs)
        docker compose -p $PROJECT_NAME logs -f ${2:-}
        ;;

    logs-tail)
        docker compose -p $PROJECT_NAME logs --tail 100 ${2:-}
        ;;

    shell)
        echo "Opening shell in app container..."
        docker exec -it ${PROJECT_NAME}-app /bin/bash
        ;;

    rebuild)
        echo "Rebuilding and redeploying..."
        docker compose -p $PROJECT_NAME down
        docker compose -p $PROJECT_NAME up -d --build
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
        docker exec ${PROJECT_NAME}-worker celery -A celery_app inspect active 2>/dev/null || echo "No workers running"
        ;;

    cleanup)
        echo "Cleaning up Docker resources..."
        docker compose -p $PROJECT_NAME down -v
        echo "Cleanup complete (volumes removed)"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs|logs-tail|shell|rebuild|test|worker-status|cleanup}"
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
