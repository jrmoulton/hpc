#!/bin/bash

# Parameters for the histogram
BIN_COUNT=100
MIN_VALUE=0.0
MAX_VALUE=100.0
DATA_COUNT=100000  # 10 million points for better scaling

# Output file for the results
OUTPUT_FILE="benchmark_results.txt"

# Clear previous results
> $OUTPUT_FILE

# Run benchmarks for thread counts 1-8
for threads in {1..8}
do
    echo "Benchmarking with $threads threads..."
    
    # Run hyperfine with 10 warmup runs and 20 timed runs
    hyperfine --warmup 10 \
              --runs 20 \
              --export-json "thread_${threads}.json" \
              --command-name "$threads threads" \
              "./target/hw3 $threads $DATA_COUNT"
              # "./target/parallel $threads $BIN_COUNT $MIN_VALUE $MAX_VALUE $DATA_COUNT"
    
    # Extract the mean time and append to results file
    mean_time=$(cat "thread_${threads}.json" | jq '.results[0].mean')
    echo "$threads,$mean_time" >> $OUTPUT_FILE
done

echo "Benchmarking complete. Results saved to $OUTPUT_FILE"
