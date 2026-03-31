#!/bin/bash
# RunPod Setup Script for SmolVLA Inference Server
#
# Steps to use:
# 1. Create a RunPod GPU Pod:
#    - Template: RunPod PyTorch (or any CUDA image)
#    - GPU: RTX 4090 (24GB) or A4000 (16GB) — cheapest options that work
#    - Expose HTTP port 8000 in pod settings
#
# 2. SSH into the pod or use the web terminal
#
# 3. Run this script:
#    bash runpod_setup.sh
#
# 4. After setup completes, the server will start automatically.
#    Note the pod's public URL (e.g., https://<POD_ID>-8000.proxy.runpod.net)
#
# 5. On your local PC, run:
#    python smolvla_client.py --server_url https://<POD_ID>-8000.proxy.runpod.net

set -e

echo "=== SmolVLA RunPod Setup ==="

# Install system deps
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git git-lfs > /dev/null 2>&1

# Clone lerobot
if [ ! -d "lerobot" ]; then
    echo "Cloning lerobot..."
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
else
    echo "lerobot already cloned, updating..."
    cd lerobot
    git pull origin main
fi

# Install lerobot with SmolVLA dependencies
echo "Installing lerobot[smolvla]..."
pip install -e ".[smolvla]" -q

# Install server dependencies
echo "Installing server dependencies..."
pip install fastapi uvicorn python-multipart pillow -q

# Set use_degrees=False for compatibility with user's dataset
echo "Setting use_degrees=False..."
sed -i 's/use_degrees: bool = True/use_degrees: bool = False/' \
    src/lerobot/robots/so_follower/config_so_follower.py
sed -i 's/use_degrees: bool = True/use_degrees: bool = False/' \
    src/lerobot/teleoperators/so_leader/config_so_leader.py

# Copy server script if not present
if [ ! -f "smolvla_server.py" ]; then
    echo "ERROR: smolvla_server.py not found. Copy it to the lerobot directory."
    echo "You can upload it via: scp smolvla_server.py runpod:~/lerobot/"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Starting SmolVLA server..."
echo "Model: AdithyaRajendran/smolvla_so101_grab_brain_t2_v9"
echo ""

python smolvla_server.py \
    --policy_repo "AdithyaRajendran/smolvla_so101_grab_brain_t2_v9" \
    --device cuda \
    --host 0.0.0.0 \
    --port 8000
