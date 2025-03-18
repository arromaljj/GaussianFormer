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
conda env create -f environment.yml || { echo "Failed to create conda environment from environment.yml"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate selfocc || { echo "Failed to activate conda environment"; exit 1; }


