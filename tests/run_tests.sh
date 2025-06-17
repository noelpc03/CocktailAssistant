#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="$SCRIPT_DIR/results"
QUERIES_FILE="$SCRIPT_DIR/sample_queries.json"

# Ensure the results directory exists
mkdir -p "$RESULTS_DIR"

# Check if the server is running
check_server() {
    echo -e "${BLUE}Checking if server is running...${NC}"
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN}Server is running.${NC}"
        return 0
    else
        echo -e "${RED}Server doesn't appear to be running.${NC}"
        return 1
    fi
}

# Run simplified test suite
run_simplified_test() {
    echo -e "${BLUE}Running simplified test suite...${NC}"
    
    if [ ! -f "$QUERIES_FILE" ]; then
        echo -e "${RED}Error: Sample queries file not found at $QUERIES_FILE${NC}"
        return 1
    fi
    
    python3 "$SCRIPT_DIR/test_simplified.py" --file "$QUERIES_FILE"
    
    echo -e "${GREEN}Simplified tests completed.${NC}"
}

# Run single query test
run_single_query() {
    echo -e "${BLUE}Enter your query:${NC}"
    read -r query
    
    echo -e "${BLUE}Running test with query: '${query}'${NC}"
    python3 "$SCRIPT_DIR/test_simplified.py" --query "$query"
    
    echo -e "${GREEN}Test completed.${NC}"
}

# Run consistency test
run_consistency_test() {
    echo -e "${BLUE}Enter your query for consistency testing:${NC}"
    read -r query
    
    echo -e "${BLUE}How many times do you want to run this query? (default: 5)${NC}"
    read -r repetitions
    
    if [ -z "$repetitions" ]; then
        repetitions=5
    fi
    
    echo -e "${BLUE}Delay between queries in seconds? (default: 2)${NC}"
    read -r delay
    
    if [ -z "$delay" ]; then
        delay=2
    fi
    
    echo -e "${BLUE}Running consistency test with query '${query}', ${repetitions} repetitions, ${delay}s delay...${NC}"
    python3 "$SCRIPT_DIR/test_consistency.py" --query "$query" --repetitions "$repetitions" --delay "$delay"
    
    echo -e "${GREEN}Consistency test completed.${NC}"
}

# List and view test results
view_results() {
    echo -e "${BLUE}Available test results:${NC}"
    
    # List all JSON files in results directory with numbers
    file_list=("$RESULTS_DIR"/*.json)
    
    if [ ${#file_list[@]} -eq 0 ] || [ ! -e "${file_list[0]}" ]; then
        echo -e "${YELLOW}No test results found.${NC}"
        return
    fi
    
    for i in "${!file_list[@]}"; do
        filename=$(basename "${file_list[$i]}")
        echo -e "$((i+1)). $filename"
    done
    
    echo ""
    echo -e "${BLUE}Enter the number of the file to view, or 0 to go back:${NC}"
    read -r selection
    
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -gt 0 ] && [ "$selection" -le "${#file_list[@]}" ]; then
        selected_file="${file_list[$((selection-1))]}"
        echo -e "${BLUE}Contents of $(basename "$selected_file"):${NC}"
        cat "$selected_file" | python3 -m json.tool
        
        echo ""
        echo -e "${BLUE}Press Enter to continue...${NC}"
        read -r
    elif [ "$selection" != "0" ]; then
        echo -e "${RED}Invalid selection.${NC}"
    fi
}

# Main menu
show_menu() {
    clear
    echo -e "${GREEN}===================================${NC}"
    echo -e "${GREEN}= Cocktail IR System Testing Tool =${NC}"
    echo -e "${GREEN}===================================${NC}"
    echo ""
    echo -e "${BLUE}1.${NC} Run simplified test suite (all sample queries)"
    echo -e "${BLUE}2.${NC} Test a single query"
    echo -e "${BLUE}3.${NC} Run consistency test (same query multiple times)"
    echo -e "${BLUE}4.${NC} View test results"
    echo -e "${BLUE}5.${NC} Exit"
    echo ""
    echo -e "${YELLOW}Please select an option:${NC}"
}

# Main function
main() {
    while true; do
        show_menu
        read -r choice
        
        case "$choice" in
            1)
                if check_server; then
                    run_simplified_test
                    echo -e "${BLUE}Press Enter to continue...${NC}"
                    read -r
                else
                    echo -e "${RED}Please start the server first.${NC}"
                    echo -e "${BLUE}Press Enter to continue...${NC}"
                    read -r
                fi
                ;;
            2)
                if check_server; then
                    run_single_query
                    echo -e "${BLUE}Press Enter to continue...${NC}"
                    read -r
                else
                    echo -e "${RED}Please start the server first.${NC}"
                    echo -e "${BLUE}Press Enter to continue...${NC}"
                    read -r
                fi
                ;;
            3)
                if check_server; then
                    run_consistency_test
                    echo -e "${BLUE}Press Enter to continue...${NC}"
                    read -r
                else
                    echo -e "${RED}Please start the server first.${NC}"
                    echo -e "${BLUE}Press Enter to continue...${NC}"
                    read -r
                fi
                ;;
            4)
                view_results
                ;;
            5)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please try again.${NC}"
                sleep 1
                ;;
        esac
    done
}

# Run the main function
main
