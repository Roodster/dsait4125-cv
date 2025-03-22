#!/bin/bash
# build_container.sh

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Catch errors in pipelines

# Define the name of your definition file and output SIF file
DEF_FILE="./apptainer/environment.def"
SIF_FILE="./apptainer/container.sif"

# Check if Apptainer is installed
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer is not installed. Please install it before running this script."
    exit 1
fi

# Remove existing SIF file to prevent build conflicts
if [ -f "$SIF_FILE" ]; then
    echo "Removing existing container: $SIF_FILE"
    rm -f "$SIF_FILE"
fi

# Build the container
echo "Building container from $DEF_FILE..."
apptainer build "$SIF_FILE" "$DEF_FILE"

# Verify the build
if [ -f "$SIF_FILE" ]; then
    echo "Container built successfully: $SIF_FILE"
else
    echo "Error: Container build failed."
    exit 1
fi