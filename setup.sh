#!/bin/bash

# Step 1: Update Jetson system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install necessary system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv libopenblas-dev ffmpeg libportaudio2

# Step 3: Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv whisper_venv
source whisper_venv/bin/activate

# Step 4: Install PyTorch (compatible with JetPack)
echo "Installing PyTorch and relevant libraries..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: If PyTorch wheel is downloaded manually from NVIDIAs site, install it
# echo "Installing manually downloaded PyTorch wheel (optional)..."
# pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# Step 5: Install Whisper and other Python dependencies
echo "Installing Whisper and other Python packages..."
pip install -r requirements.txt