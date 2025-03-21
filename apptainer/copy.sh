#!/bin/bash

# SFTP details
SFTP_HOST="sftp.tudelft.nl"
SFTP_USER="rhaket"
SFTP_PASSWORD="S#Fw$Ro92Q4fM&St9y#p"
SFTP_PORT=22
REMOTE_DIR="/staff-umbrella/rhaket/repository"

# Local repository details
LOCAL_REPO="./"

# Export the password to avoid showing it in the command line
export SSHPASS="$SFTP_PASSWORD"

# Find and copy specific files to SFTP while maintaining folder structure and ignoring .venv
echo "Finding and copying files to SFTP..."

find "$LOCAL_REPO" -type d -path "*/.venv" -prune -o -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.xml" \) -print0 | while IFS= read -r -d '' file; do
    # Calculate the relative path from LOCAL_REPO to the file
    rel_path=$(realpath --relative-to="$LOCAL_REPO" "$file")
    remote_path="$REMOTE_DIR/$(dirname "$rel_path")"

    # Create the remote directory if it does not exist
    sshpass -e ssh $SFTP_USER@$SFTP_HOST "mkdir -p $remote_path"

    # Upload the file using sshpass with sftp
    echo "Uploading $file to SFTP..."
    sshpass -e sftp $SFTP_USER@$SFTP_HOST <<< $"put $file $remote_path/"
done

echo "File transfer completed."