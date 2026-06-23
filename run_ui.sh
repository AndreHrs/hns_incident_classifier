#!/bin/bash

set -e

MARKER=".installed"

echo "============================================"
echo "  Incident Classifier — Web UI"
echo "============================================"

if [ ! -f "$MARKER" ]; then
    echo "[Run] Not installed yet — running installer first..."
    echo ""
    ./install.sh
    echo ""
fi

# Read install metadata
ENV_TYPE=$(grep "^env_type=" "$MARKER" | cut -d= -f2)
ENV_NAME=$(grep "^env_name=" "$MARKER" | cut -d= -f2)
GPU=$(grep "^gpu=" "$MARKER" | cut -d= -f2)

echo "[Run] GPU backend  : $GPU"
echo "[Run] Environment  : $ENV_TYPE ($ENV_NAME)"

# Activate environment
if [ "$ENV_TYPE" = "conda" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    source .venv/bin/activate
fi

if [ -f .env ]; then
    set -a; source .env; set +a
fi

HEADLESS=""
for arg in "$@"; do
    case $arg in
        --headless) HEADLESS="--server.headless true --server.port 8501" ;;
    esac
done

echo "[Run] Starting Streamlit app..."
echo ""
# shellcheck disable=SC2086
streamlit run app/app.py $HEADLESS
