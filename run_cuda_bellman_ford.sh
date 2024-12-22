#!/bin/bash

# Define your executable and data folder
EXEC="./cuda_bellman_ford"  # Path to your compiled CUDA program
DATAFOLDER="./bellman_ford_data"        # Path to folder containing input files
OUTPUTDATAFOLDER="./cuda_bellman_ford_results"
NUMTHREADS=${1:-256}                     # Number of threads (example)

# Loop over input files (assuming they are named input1.txt, input2.txt, etc.)
for i in {1..10}; do  # Modify {1..10} to the range of input files you have

    # Define input and expected output files
    INPUT="${DATAFOLDER}/input${i}.txt"
    EXPECTED_OUTPUT="${DATAFOLDER}/output${i}.txt"
    OUTPUT="${OUTPUTDATAFOLDER}/output${i}.txt"  # The file where cuda_bellman_ford_c_v3 writes the result
    
    echo ""
    # Run the program
    echo "Running ${EXEC} on ${INPUT}..."
    $EXEC $INPUT $NUMTHREADS 

    # Compare the program's output with the expected output
    echo "Comparing results for input${i}..."
    if diff -b "$OUTPUT" "$EXPECTED_OUTPUT" >/dev/null; then
        echo "Test ${i} passed! The outputs match."
    else
        echo "Test ${i} failed! Outputs do not match."
    fi
done
