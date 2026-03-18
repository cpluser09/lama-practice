#!/bin/bash
# 快速重载服务代码（无需重新构建镜像）
# 用于修改网页或服务代码后的快速更新

echo "Restarting container to reload code..."
docker-compose restart lama-service

echo "Waiting for service to be ready..."
sleep 3

echo "Checking service status..."
docker-compose ps

echo ""
echo "Service reloaded! Access at: http://localhost:5001"
echo "View logs: docker-compose logs -f"
