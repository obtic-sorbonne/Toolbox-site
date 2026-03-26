#!/bin/bash

# Configuration
IMAGE_NAME="pandore:test"
CONTAINER_NAME="pandore_local_test"
PORT="${1:-8552}"  # default port 8552, but accepts an argument

# Build the image
echo "Building $IMAGE_NAME..."
sudo docker build -t $IMAGE_NAME .

# Stop and remove existing container if it exists
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}_${PORT}$"; then
    echo "Stopping existing container on port $PORT..."
    sudo docker stop ${CONTAINER_NAME}_${PORT}
    sudo docker rm ${CONTAINER_NAME}_${PORT}
fi

# Start the new container
echo "Starting container on port $PORT..."
sudo docker run -d \
    --name ${CONTAINER_NAME}_${PORT} \
    -p ${PORT}:5000 \
    --gpus all \
    --runtime=nvidia \
    -e PYTHONUNBUFFERED=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e SCRIPT_NAME= \
    --memory="16g" \
    --memory-reservation="8g" \
    --cpus=2 \
    $IMAGE_NAME \
    conda run --no-capture-output -n toolbox_env gunicorn --workers 4 --timeout 600 --bind 0.0.0.0:5000 toolbox_app:app

echo "Done! App running at localhost:${PORT}"