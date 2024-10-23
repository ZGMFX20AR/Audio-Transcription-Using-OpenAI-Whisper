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

# Step 4.1
# Optional: Uncomment this section if you are using another version of Jetpack, download PyTorch wheel from NVIDIAs site, install it (https://developer.nvidia.com/embedded/downloads#?search=pytorch)
# echo "Installing manually downloaded PyTorch wheel (optional)..."
# Example pip install >name of the wheel file<.whl
# pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Step 5: Install Whisper and other Python dependencies
echo "Installing Whisper and other Python packages..."
pip install -r requirements.txt