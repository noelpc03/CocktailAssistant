#!/usr/bin/env python3
"""
Consistency test script for the cocktail information retrieval system.
Runs the same query multiple times to verify if responses are stable.
Uses agent_system directly instead of HTTP requests.
"""

import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
import os
import statistics
import re

# Constants
AGENT_SYSTEM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_system.sh")
DEFAULT_REPETITIONS = 5

def calculate_stats(response_times):
    """Calculate statistics for a list of response times."""
    if not response_times:
        return {}
    
    return {
        "min": min(response_times),
        "max": max(response_times),
        "mean": statistics.mean(response_times),
        "median": statistics.median(response_times),
        "stdev": statistics.stdev(response_times) if len(response_times) > 1 else 0
    }

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

def run_query(query):
    """Run a single query and return results."""
    start_time = time.time()
    try:
        # Run agent_system.sh search with the query
        command = f'bash "{AGENT_SYSTEM_PATH}" search "{query}"'
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        
        stdout, stderr = process.communicate()
        output = stdout + stderr
        end_time = time.time()
        
        # Extract information from output
        strategy, crawler_used, response = extract_info_from_output(output)
        
        return {
            "status": "success",
            "strategy": strategy,
            "crawler_used": crawler_used,
            "response": response,
            "response_time": end_time - start_time,
            "raw_output": output
        }
    except Exception as e:
        return {
            "status": "exception",
            "error": str(e),
            "response_time": time.time() - start_time
        }

def run_consistency_test(query, repetitions=DEFAULT_REPETITIONS, delay_seconds=2):
    """Run the same query multiple times and evaluate consistency."""
    print(f"Running consistency test for query: {query}")
    print(f"Repetitions: {repetitions}, Delay between queries: {delay_seconds} seconds")
    
    results = []
    strategies = set()
    crawler_usage = []
    response_times = []
    responses = []
    
    for i in range(repetitions):
        print(f"Running iteration {i+1}/{repetitions}...")
        result = run_query(query)
        results.append(result)
        
        if result["status"] == "success":
            strategies.add(result.get("strategy", "unknown"))
            crawler_usage.append(result.get("crawler_used", False))
            response_times.append(result["response_time"])
            responses.append(result.get("response", ""))
        
        # Wait between requests
        if i < repetitions - 1:  # Don't wait after last query
            time.sleep(delay_seconds)
    
    # Calculate statistics
    time_stats = calculate_stats(response_times)
    
    # Calculate strategy consistency
    strategy_consistent = len(strategies) == 1 if strategies else False
    dominant_strategy = list(strategies)[0] if len(strategies) == 1 else "inconsistent"
    
    # Calculate crawler usage consistency
    crawler_usage_consistent = all(x == crawler_usage[0] for x in crawler_usage) if crawler_usage else False
    
    # Simple calculation of response similarity (check if all responses are exactly the same)
    # In a more advanced version, you could use similarity metrics like cosine similarity
    responses_identical = len(set(responses)) == 1 if responses else False
    
    # Count status types
    success_count = sum(1 for r in results if r.get("status") == "success")
    timeout_count = sum(1 for r in results if r.get("status") == "timeout")
    error_count = sum(1 for r in results if r.get("status") in ["error", "exception"])
    
    return {
        "query": query,
        "repetitions": repetitions,
        "success_count": success_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "strategy_consistent": strategy_consistent,
        "dominant_strategy": dominant_strategy,
        "strategies_used": list(strategies),
        "crawler_usage_consistent": crawler_usage_consistent, 
        "crawler_usage_counts": {
            "true": sum(1 for x in crawler_usage if x),
            "false": sum(1 for x in crawler_usage if not x)
        },
        "responses_identical": responses_identical,
        "time_statistics": time_stats,
        "individual_results": results
    }

def main():
    parser = argparse.ArgumentParser(description="Run consistency tests for the cocktail IR system")
    parser.add_argument("--query", required=True, help="Query to test repeatedly")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, 
                        help=f"Number of times to run the query (default: {DEFAULT_REPETITIONS})")
    parser.add_argument("--delay", type=int, default=2, 
                        help="Delay in seconds between queries (default: 2)")
    parser.add_argument("--output", help="Output file for results (defaults to timestamp-based filename)")
    
    args = parser.parse_args()
    
    # Run consistency test
    results = run_consistency_test(args.query, args.repetitions, args.delay)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output if args.output else f"tests/results/consistency_test_{timestamp}.json"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nConsistency test completed. Results saved to {output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Query: '{args.query}'")
    print(f"Success rate: {results['success_count']}/{args.repetitions} ({results['success_count']/args.repetitions*100:.1f}%)")
    print(f"Strategy consistent: {results['strategy_consistent']} (dominant: {results['dominant_strategy']})")
    print(f"Crawler usage consistent: {results['crawler_usage_consistent']}")
    print(f"Responses identical: {results['responses_identical']}")
    
    if results['time_statistics']:
        print(f"Response time (seconds): min={results['time_statistics']['min']:.2f}, " +
              f"max={results['time_statistics']['max']:.2f}, " +
              f"mean={results['time_statistics']['mean']:.2f}, " +
              f"median={results['time_statistics']['median']:.2f}, " +
              f"stdev={results['time_statistics']['stdev']:.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
