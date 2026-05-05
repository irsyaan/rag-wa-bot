#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/rag-wa-bot"
IMAGE_NAME="personal-wa-assistant:local"
CONTAINER_NAME="personal-wa-assistant"

MODE="${1:-proxy}"

if [[ "$MODE" != "proxy" && "$MODE" != "normal" ]]; then
  echo "Usage: ./deploy.sh proxy|normal"
  exit 1
fi

cd "$APP_DIR"

echo "============================================"
echo "Deploying rag-wa-bot"
echo "Mode: $MODE"
echo "Directory: $APP_DIR"
echo "============================================"

echo
echo "Checking required files..."

if [[ "$MODE" == "proxy" ]]; then
  COMPOSE_FILE="docker-compose.proxy.yml"
  ENV_FILE=".env.proxy"
else
  COMPOSE_FILE="docker-compose.yml"
  ENV_FILE=".env"
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Missing $COMPOSE_FILE"
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  exit 1
fi

if [[ ! -f "requirements.txt" ]]; then
  echo "Missing requirements.txt"
  exit 1
fi

if [[ ! -f "Dockerfile" ]]; then
  echo "Missing Dockerfile"
  exit 1
fi

echo "Required files OK"

echo
echo "Pulling latest code from Git..."
git pull --ff-only

echo
echo "Preparing wheelhouse..."
rm -rf wheelhouse
mkdir -p wheelhouse

python3 -m pip download \
  --dest wheelhouse \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  -r requirements.txt

echo
echo "Building Docker image..."
docker build --no-cache -t "$IMAGE_NAME" .

echo
echo "Restarting container with $COMPOSE_FILE..."
docker compose -f "$COMPOSE_FILE" down || true
docker compose -f "$COMPOSE_FILE" up -d --force-recreate

echo
echo "Container status:"
docker ps -a | grep "$CONTAINER_NAME" || true

echo
echo "Verifying running code markers:"
docker exec "$CONTAINER_NAME" grep -n "RAG found\|Qdrant search\|_extract_points" /app/app/rag_engine.py || true

echo
echo "Latest logs:"
docker logs "$CONTAINER_NAME" --tail 80

echo
echo "============================================"
echo "Deploy finished"
echo "To follow logs:"
echo "docker logs -f $CONTAINER_NAME"
echo "============================================"