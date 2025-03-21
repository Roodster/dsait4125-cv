#!/bin/bash
#SBATCH --job-name=drac_training
#SBATCH --output=drac_training_%j.log
#SBATCH --time=01:00:00  # 1 hour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# Ensure outputs directory exists
mkdir -p ./outputs

# Run the container with proper bind mounts and pass arguments
apptainer run --nv \
    --bind $(pwd):/src \
    --bind $(pwd)/outputs:/outputs \
    ./apptainer/container.sif "$@"