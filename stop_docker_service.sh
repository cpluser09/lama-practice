#!/bin/bash
# Stop LaMa Docker Service

echo "Stopping LaMa Docker service..."

# Stop docker-compose
docker-compose down

echo "Docker service stopped."
