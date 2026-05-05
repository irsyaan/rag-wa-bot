#!/usr/bin/env bash
set -euo pipefail

cd /opt/rag-wa-bot

git pull --ff-only

docker build -t personal-wa-assistant:local .

docker compose -f docker-compose.proxy.yml up -d --force-recreate

docker logs personal-wa-assistant --tail 80