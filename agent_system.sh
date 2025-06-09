#!/bin/bash

# Script for running the multiagent bartender information retrieval system

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set Python path
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Check if the command is "crawl" and add --wait option automatically
if [[ "$1" == "crawl" ]]; then
    # Check if --wait is already in the arguments
    if ! [[ "$*" =~ "--wait" ]]; then
        # Add --wait to the arguments
        python3 "$(dirname "$SCRIPT_DIR")/ia-sri-sim/agents/main.py" "$@" --wait
    else
        # Run with existing arguments
        python3 "$(dirname "$SCRIPT_DIR")/ia-sri-sim/agents/main.py" "$@"
    fi
else
    # For other commands, run normally
    python3 "$(dirname "$SCRIPT_DIR")/ia-sri-sim/agents/main.py" "$@"
fi
