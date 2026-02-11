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

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="pdfx"

echo "=== PDF Extraction Service Deployment ==="

# Step 1: Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Copy from .env.example and configure."
    exit 1
fi

# Step 2: Create persistent directories
echo "Creating persistent directories..."
mkdir -p data/cache
mkdir -p data/uploads
mkdir -p data/models
mkdir -p logs

# Step 3: Stop existing services
echo "Stopping existing services..."
docker compose -p $PROJECT_NAME down 2>/dev/null || true

# Step 4: Build and start services
echo "Building and starting services..."
docker compose -p $PROJECT_NAME up -d --build

# Step 5: Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Step 6: Health checks
echo ""
echo "=== Health Checks ==="

# Check GROBID
echo -n "  GROBID: "
if curl -sf http://localhost:8070/api/isalive > /dev/null 2>&1; then
    echo "HEALTHY"
else
    echo "NOT READY (may still be loading models, check: docker compose -p $PROJECT_NAME logs grobid)"
fi

# Check Redis
echo -n "  Redis: "
if docker exec ${PROJECT_NAME}-redis-1 redis-cli ping 2>/dev/null | grep -q PONG; then
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

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Service endpoints:"
echo "  Web UI:    http://localhost:5000"
echo "  REST API:  http://localhost:5000/api/v1/"
echo "  Health:    http://localhost:5000/api/v1/health"
echo "  GROBID:    http://localhost:8070"
echo ""
echo "Management:"
echo "  Logs:      docker compose -p $PROJECT_NAME logs -f"
echo "  Status:    ./manage.sh status"
echo "  Stop:      docker compose -p $PROJECT_NAME down"
