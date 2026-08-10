#!/bin/bash
# Clean up script for sensor loop workflow

WORKSPACE=${1:-.}

echo "Cleaning up generated files and subdirectories in ${WORKSPACE}, but keeping plots and .mh data..."

# Remove subdirectories
rm -rf "${WORKSPACE}"/sensor_case-* "${WORKSPACE}"/sensor_initial_state

# Remove mesh and log files
rm -f "${WORKSPACE}"/*.npz
rm -f "${WORKSPACE}"/*.vtu
rm -f "${WORKSPACE}"/*.csv
rm -f "${WORKSPACE}"/params.log
rm -f "${WORKSPACE}"/sensor_loop_*.log

echo "Clean up complete for ${WORKSPACE}."
