#!/bin/bash

# Script for running the multiagent bartender information retrieval system
set -x  # Enable debug mode to print each command

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Set Python path
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Command: $@"

# Check if main.py exists
if [ ! -f "$PROJECT_ROOT/agents/main.py" ]; then
    echo "ERROR: main.py not found at $PROJECT_ROOT/agents/main.py"
    exit 1
fi

# Command to extract ontology triples (useful for testing)
if [[ "$1" == "extract_ontology" ]]; then
    echo "Ejecutando extracción de ontología..."
    python3 "$PROJECT_ROOT/agents/main.py" ontology extract
# Command to test the strategy agent
elif [[ "$1" == "test_strategy" ]]; then
    echo "Ejecutando prueba del agente de estrategia..."
    # Instalar dependencias si es necesario
    echo "Verificando dependencias..."
    pip install numpy requests > /dev/null 2>&1
    # Ejecutar la prueba
    python3 "$PROJECT_ROOT/test_strategy_agent.py"
# Check if the command is "crawl" and add --wait option automatically
elif [[ "$1" == "crawl" ]]; then
    # Check if --wait is already in the arguments
    if ! [[ "$*" =~ "--wait" ]]; then
        # Add --wait to the arguments
        python3 "$PROJECT_ROOT/agents/main.py" "$@" --wait
    else
        # Run with existing arguments
        python3 "$PROJECT_ROOT/agents/main.py" "$@"
    fi
else
    # For other commands, run normally
    echo "Running: python3 $PROJECT_ROOT/agents/main.py $@"
    # Use unbuffer if available to force line buffering
    if command -v unbuffer &> /dev/null; then
        unbuffer python3 "$PROJECT_ROOT/agents/main.py" "$@" 2>&1 | tee ./agent_system_output.log
        EXIT_CODE=${PIPESTATUS[0]}
    else
        # If unbuffer is not available, use stdbuf as an alternative
        if command -v stdbuf &> /dev/null; then
            stdbuf -oL python3 "$PROJECT_ROOT/agents/main.py" "$@" 2>&1 | tee ./agent_system_output.log
            EXIT_CODE=${PIPESTATUS[0]}
        else
            # As a last resort, use python with line buffering
            PYTHONUNBUFFERED=1 python3 "$PROJECT_ROOT/agents/main.py" "$@" 2>&1 | tee ./agent_system_output.log
            EXIT_CODE=${PIPESTATUS[0]}
        fi
    fi
    
    echo "Command output saved to agent_system_output.log"
    # Also display just the query results from the log file
    echo "----------------------------------------"
    echo "QUERY RESULTS:"
    echo "----------------------------------------"
    grep -A 100 "=== QUERY RESULTS ===" ./agent_system_output.log | tail -n +2
    echo "----------------------------------------"
    echo "Command completed with exit code: $EXIT_CODE"
fi
