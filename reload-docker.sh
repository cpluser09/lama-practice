#!/bin/bash
# Docker 服务快速重载 (无需重建镜像)
# Quick reload Docker service without rebuild

set -e

echo "Reloading Docker service..."
docker-compose restart lama-service

echo "Waiting for service..."
sleep 3

echo "✓ Ready at: http://localhost:5001"
echo "  Logs: docker-compose logs -f"
