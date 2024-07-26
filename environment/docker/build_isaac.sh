#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

docker build --progress=plain --network host -t isaac -f docker/Dockerfile_Isaacgym .