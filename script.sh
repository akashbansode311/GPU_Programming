#!/bin/bash
#SBATCH --job-name=CUDA                  # Job name
#SBATCH --output=cuda_%j.out             # Output file
#SBATCH --error=cuda_%j.err              # Error log
#SBATCH --partition=gpu                  # Partition
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --time=00:30:00                  # Max time (hh:mm:ss)

# Load CUDA module (adjust according to your system)
#module load cuda-11.2
. /home/apps/spack/share/spack/setup-env.sh
spack load cuda/snntalj

cd $HOME/GPU_Programming

# Run the CUDA executable
time ./filename
