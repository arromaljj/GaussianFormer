#!/bin/bash
set -e

echo "=== Setting up environment for GaussianFormer ==="

# Check if script is run from the correct directory
if [ ! -d "model" ]; then
    echo "Error: This script must be run from the GaussianFormer root directory!"
    echo "Please navigate to the root directory and try again."
    exit 1
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
fi

# 1. Create and activate conda environment
echo "=== Creating conda environment ==="
conda create -n selfocc python=3.8.16 -y || { echo "Failed to create conda environment"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate selfocc || { echo "Failed to activate conda environment"; exit 1; }

# 2. Install PyTorch
echo "=== Installing PyTorch ==="
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 || { echo "Failed to install PyTorch"; exit 1; }

# 3. Install packages from MMLab
echo "=== Installing MMLab packages ==="
pip install openmim || { echo "Failed to install openmim"; exit 1; }
mim install mmcv==2.0.1 || { echo "Failed to install mmcv"; exit 1; }
mim install mmdet==3.0.0 || { echo "Failed to install mmdet"; exit 1; }
mim install mmsegmentation==1.0.0 || { echo "Failed to install mmsegmentation"; exit 1; }
mim install mmdet3d==1.1.1 || { echo "Failed to install mmdet3d"; exit 1; }

# 4. Install other packages
echo "=== Installing other dependencies ==="
pip install spconv-cu117 || { echo "Failed to install spconv-cu117"; exit 1; }
pip install timm || { echo "Failed to install timm"; exit 1; }
pip install einops || { echo "Failed to install einops"; exit 1; }
pip install jaxtyping || { echo "Failed to install jaxtyping"; exit 1; }

# 5. Install custom CUDA ops
echo "=== Installing custom CUDA operations ==="
# Create ckpts directory if it doesn't exist
mkdir -p ckpts

# Gaussian encoder ops
if [ -d "model/encoder/gaussian_encoder/ops" ]; then
    cd model/encoder/gaussian_encoder/ops
    pip install -e . || { echo "Failed to install gaussian encoder ops"; exit 1; }
    cd ../../../../
else
    echo "Warning: Directory model/encoder/gaussian_encoder/ops not found"
fi

# Local aggregation ops
if [ -d "model/head/localagg" ]; then
    cd model/head/localagg
    pip install -e . || { echo "Failed to install localagg ops"; exit 1; }
    cd ../../../
else
    echo "Warning: Directory model/head/localagg not found"
fi

# For GaussianFormer-2
if [ -d "model/head/localagg_prob" ]; then
    cd model/head/localagg_prob
    pip install -e . || { echo "Failed to install localagg_prob ops"; exit 1; }
    cd ../../../
else
    echo "Warning: Directory model/head/localagg_prob not found"
fi

if [ -d "model/head/localagg_prob_fast" ]; then
    cd model/head/localagg_prob_fast
    pip install -e . || { echo "Failed to install localagg_prob_fast ops"; exit 1; }
    cd ../../../
else
    echo "Warning: Directory model/head/localagg_prob_fast not found"
fi

# 6. Install visualization packages (optional)
echo "=== Installing visualization packages ==="
pip install pyvirtualdisplay mayavi matplotlib==3.7.2 PyQt5 || { echo "Warning: Failed to install some visualization packages"; }

# 7. Download pretrained weights for image backbone
echo "=== Downloading pretrained weights ==="
mkdir -p ckpts
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth -O ckpts/r101_dcn_fcos3d_pretrain.pth || { echo "Warning: Failed to download pretrained weights"; }

# 8. Add automatic activation of selfocc environment to .bashrc
echo "=== Adding automatic selfocc environment activation to .bashrc ==="
if grep -q "conda activate selfocc" ~/.bashrc; then
    echo "Environment activation already in .bashrc"
else
    echo -e "\n# Automatically activate selfocc environment\nconda activate selfocc" >> ~/.bashrc
    echo "Added environment activation to .bashrc"
fi

echo "=== Setup completed successfully! ==="
echo "=== The selfocc environment will be automatically activated in new terminal sessions ==="
echo "=== To activate it now, run: source ~/.bashrc ==="