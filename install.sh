#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="deepbranchai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_conda() {
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        printf '%s\n' "${CONDA_EXE}"
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    local candidate
    for candidate in \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "/opt/conda/bin/conda"
    do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

echo "=== DeepBranchAI Installation ==="

CONDA_BIN="$(find_conda)" || {
    echo "ERROR: conda not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
}
echo "Found conda at: $CONDA_BIN"

CONDA_BASE="$("$CONDA_BIN" info --base)"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Could not find conda.sh under $CONDA_BASE" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

cd "$SCRIPT_DIR"

echo
echo "Checking conda environment '$ENV_NAME'..."
if conda run -n "$ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
    echo "Reusing existing conda environment '$ENV_NAME'."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.12 -y
fi

conda activate "$ENV_NAME"

echo
echo "Active Python:"
python -c "import sys; print(sys.executable)"

echo
echo "Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing PyTorch with CUDA 11.8..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo
echo "Installing DeepBranchAI dependencies..."
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME"

echo
echo "Verifying installation..."
python -c "import torch, nnunetv2, nibabel, jupyter; print('PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available()); print('nnU-Net v2 OK'); print('nibabel OK'); print('jupyter OK')"

echo
echo "=== Installation complete ==="
echo
echo "To use DeepBranchAI:"
echo "  conda activate $ENV_NAME"
echo "  jupyter notebook"
