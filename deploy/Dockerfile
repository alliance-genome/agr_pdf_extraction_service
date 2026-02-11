# Dockerfile for agr_pdf_extraction_service
# CPU-first deployment -- no GPU required
# Adapted from ai_curation_prototype/backend/docling-service/Dockerfile

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only first (keeps image smaller, no CUDA)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install production server and task queue
RUN pip install --no-cache-dir \
    gunicorn \
    celery[redis] \
    redis

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/cache data/uploads data/models logs

EXPOSE 5000

# Suppress GPU warnings
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_DEVICE=cpu
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

# Default: run the web app (override in docker-compose for worker)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "600", "app:create_app()"]
