#!/bin/bash
# sync_to_cluster.sh

# Configuration
REMOTE_USER="jacopoparretti"
REMOTE_HOST="157.27.31.122"
REMOTE_PATH="~/ssh_cluster"  # Adjust this to your remote path

# Local path (adjust if needed)
LOCAL_PATH="/Users/jacopo/Desktop/CityLearnRL-Bioinformatica/ssh_cluster/"

# Rsync command
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='logs/' \
    --exclude='models/' \
    --delete \
    -e ssh \
    "$LOCAL_PATH" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo -e "\nSync completed!"
echo -e "To run the setup on the remote machine:"
echo -e "ssh $REMOTE_USER@$REMOTE_HOST 'cd $REMOTE_PATH && ./setup.sh'"