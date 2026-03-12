#!/bin/bash
docker run -d --name pandore-prod -p 8551:5000 --gpus all --runtime=nvidia -e PYTHONUNBUFFERED=1 -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics pandore:latest
