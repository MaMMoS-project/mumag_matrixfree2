#!/bin/bash

WORKSPACE=${1:-}

if [ -z "$WORKSPACE" ]; then
    echo "Error: You must specify a workspace directory to wipe out."
    echo "Usage: ./clean_all.sh <workspace_dir>"
    echo "Example: ./clean_all.sh run_184687"
    exit 1
fi

if [ "$WORKSPACE" = "." ] || [ "$WORKSPACE" = "/" ]; then
    echo "Error: Refusing to wipe the current or root directory."
    exit 1
fi

echo "Wiping out entire workspace directory: ${WORKSPACE}..."
rm -rf "${WORKSPACE}"

# Extract Job ID from WORKSPACE string (e.g., "run_184687" or "run_cpu_184691")
JOB_ID=$(echo "$WORKSPACE" | grep -oE '[0-9]+' | tail -n 1)

if [ -n "$JOB_ID" ]; then
    echo "Removing associated log files for Job ID ${JOB_ID}..."
    rm -f *"${JOB_ID}".log
fi

echo "Workspace and logs wiped out."
