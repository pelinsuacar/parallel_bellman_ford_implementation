#!/bin/bash
#SBATCH --job-name=bellman_ford_project
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelinsu.acar@studio.unibo.it
#SBATCH --time=01:00:00           # Set the time limit (2 hours in this case)
#SBATCH --nodes=1                 # Use 1 node
#SBATCH --ntasks=1                # Run 1 task (no MPI, just 1 instance of main.sh)
#SBATCH --cpus-per-task=2         # Number of CPU cores to allocate for OpenMP
#SBATCH --gres=gpu:1              # Allocate 1 GPU (for CUDA)
#SBATCH --output=output.txt       # Log output will be written to output.txt

# Run the main script in CUDA mode
echo "Running CUDA version..."
bash ./run_cuda_bellman_ford.sh 128

# Run the main script in OpenMP mode
echo "Running OpenMP version..."
bash ./run_openmp_bellman_ford.sh 2 static 128
