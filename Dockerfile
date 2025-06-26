# Stage 1 — Build
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential cmake libpq-dev libboost-all-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    gfortran curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2 — Runtime only
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libpq-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    iproute2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH="/app"
EXPOSE 8765
ENTRYPOINT ["python", "main.py"]
