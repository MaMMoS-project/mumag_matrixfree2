#!/bin/bash
cd "$(dirname "$0")"

echo "Cleaning evaluation directories..."

if [ -d "evaluations" ]; then
    # find and remove all directories inside evaluations/ that start with eval_
    find evaluations -maxdepth 1 -type d -name "eval_*" -exec rm -rf {} +
    echo "Removed evaluation subdirectories."
else
    echo "No evaluations directory found."
fi

echo "Done. (CSV results are preserved in evaluations/)"
