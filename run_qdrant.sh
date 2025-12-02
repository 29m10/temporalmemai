#!/bin/bash
# Script to run Qdrant server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
QDRANT_BIN="${SCRIPT_DIR}/qdrant"

# Check if qdrant binary exists
if [ ! -f "$QDRANT_BIN" ]; then
    echo "Error: Qdrant binary not found at $QDRANT_BIN"
    echo "Please download it first or install Docker."
    exit 1
fi

# Run Qdrant on port 6333 (default)
echo "Starting Qdrant server on http://localhost:6333"
echo "Press Ctrl+C to stop"
"$QDRANT_BIN"

