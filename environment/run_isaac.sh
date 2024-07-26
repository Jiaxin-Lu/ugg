#!/bin/bash
set -e
set -u

container_name="isaac"  # Specify the name of the Docker container

# Check if the Docker container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}\$"; then
    # Docker container exists, attach to it
    docker exec -it "$container_name" /bin/bash
else
    # Docker container does not exist, run a new container
    docker run -it -u $(id -u):$(id -g) --privileged=True --shm-size=16g --network=host --gpus=all --name=isaac isaac /bin/bash
fi
