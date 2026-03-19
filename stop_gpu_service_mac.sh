#!/bin/bash
# Stop LaMa GPU Service on macOS

echo "Stopping LaMa GPU service..."

# Find and kill the process
PID=$(ps aux | grep "launch_gpu_service_mac.sh" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "Killing process $PID..."
    kill $PID
    echo "GPU service stopped."
else
    echo "No running GPU service found."
fi
