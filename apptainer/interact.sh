#!/bin/bash
apptainer shell --nv --bind $(pwd):/app,./outputs/:/outputs ./apptainer/container.sif
