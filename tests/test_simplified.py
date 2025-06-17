#!/usr/bin/env python3
"""
Simplified test script for the cocktail information retrieval system.
Tests strategy selection, crawler fallback, response generation, and response time.
Uses agent_system directly instead of HTTP requests.
"""

import sys
import json
import time
import argparse
from datetime import datetime
import os
import subprocess
import re

# Constants
AGENT_SYSTEM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_system.sh")
print(f"Agent system path: {AGENT_SYSTEM_PATH}")

def extract_info_from_output(output):
    """Extract strategy, crawler usage, and response from agent system output."""
    strategy = "unknown"
    crawler_used = False
    response = ""
    
    # Try to extract strategy
    strategy_match = re.search(r"Strategy: (\w+)", output)
    if strategy_match:
        strategy = strategy_match.group(1)
        
    # Alternative strategy detection from coordinator logs
    if "[Coordinator] Using ontology-based search" in output:
        strategy = "ontology"
    elif "[Coordinator] Using vector-based search" in output or "Using embedding-based search" in output:
        strategy = "embedding"
        
    # Check if crawler was used
    if "[Coordinator] Intentando usar crawler" in output or "Crawler encontró" in output:
        crawler_used = True
    
    # Extract response - look for the section between "Results for:" and any ending delimiter
    results_section = re.search(r"Results for:.*?\n-+\n(.*?)(?:-+\n\+|\Z)", output, re.DOTALL)
    if results_section:
        response = results_section.group(1).strip()
    
    return strategy, crawler_used, response

def run_test(query, test_name=None):
    """Run a test query and return results with metrics."""
    print(f"Testing query: {query}")
    
    start_time = time.time()
    try:
        # Run agent_system.sh search with the query
        command = f'bash "{AGENT_SYSTEM_PATH}" search "{query}"'
        print(f"Executing command: {command}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        
        try:
            stdout, stderr = process.communicate()  # Sin timeout
            output = stdout + stderr
            end_time = time.time()
            
            # Extract information from output
            strategy, crawler_used, response = extract_info_from_output(output)
            
            # Check if response was generated
            response_generated = response.strip() != ""
            
            # Calculate response time
            response_time = end_time - start_time
            
            # Print the response for each query
            print("\n" + "="*50)
            print(f"RESPONSE FOR: {query}")
            print("-"*50)
            print(f"Strategy: {strategy}")
            print(f"Crawler used: {crawler_used}")
            print(f"Response time: {response_time:.2f} seconds")
            print("-"*50)
            print(response)
            print("="*50 + "\n")
            
            return {
                "query": query,
                "test_name": test_name or query[:30],
                "status": "success",
                "strategy": strategy,
                "crawler_used": crawler_used,
                "response_generated": response_generated,
                "response_time": response_time,
                "response": response,
                "raw_output": output
            }
        except Exception as e:
            process.kill()
            return {
                "query": query,
                "test_name": test_name or query[:30],
                "status": "error",
                "error": str(e),
                "response_time": time.time() - start_time
            }
    except Exception as e:
        return {
            "query": query,
            "test_name": test_name or query[:30],
            "status": "exception",
            "error": str(e),
            "response_time": time.time() - start_time
        }

def run_tests_from_file(file_path):
    """Run tests from a JSON file with test queries."""
    try:
        with open(file_path, 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading test file: {e}")
        return []

    results = []
    total_tests = len(test_cases)
    
    print(f"\nRunning {total_tests} tests from file: {file_path}")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        query = test_case.get("query", "")
        test_name = test_case.get("name", None)
        
        if not query:
            continue
        
        print(f"\n[Test {i+1}/{total_tests}] Running: {test_name or query}")
        
        result = run_test(query, test_name)
        results.append(result)
        
        status = result.get("status", "unknown")
        time_taken = result.get("response_time", 0)
        print(f"Status: {status} | Time taken: {time_taken:.2f} seconds")
        print("-" * 50)
        
        # Short pause between tests
        time.sleep(2)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run simplified tests for the cocktail IR system")
    parser.add_argument("--query", help="Single query to test")
    parser.add_argument("--file", help="JSON file with test queries")
    parser.add_argument("--output", help="Output file for results (defaults to timestamp-based filename)")
    
    args = parser.parse_args()
    
    if not args.query and not args.file:
        parser.error("Either --query or --file must be specified")
    
    results = []
    
    if args.query:
        results.append(run_test(args.query))
    
    if args.file:
        file_results = run_tests_from_file(args.file)
        results.extend(file_results)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output if args.output else f"tests/results/simplified_test_{timestamp}.json"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test completed. Results saved to {output_file}")
    
    # Print summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    timeout_count = sum(1 for r in results if r.get("status") == "timeout")
    error_count = sum(1 for r in results if r.get("status") in ["error", "exception"])
    
    print(f"\nSummary:")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Timeouts: {timeout_count}")
    print(f"Errors: {error_count}")
    
    if success_count > 0:
        avg_time = sum(r.get("response_time", 0) for r in results if r.get("status") == "success") / success_count
        print(f"Average response time: {avg_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
