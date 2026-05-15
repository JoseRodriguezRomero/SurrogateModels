#!/bin/bash
#SBATCH -c 32                 # Number of Cores per MPI Task
#SBATCH --ntasks 1            # Number of MPI Tasks
#SBATCH -N 1                  # Number of Nodes
#SBATCH --mem=128G            # Requested Memory Per Node
#SBATCH -p cpu                # Partition
#SBATCH -t 40:00:00           # Job time limit
#SBATCH -o output_logfile.out # Output file

cd /home/jrodriguesro_umass_edu/Documents/SurrogateModels
julia SetupQSavory.jl
julia DistributedRunSimulations.jl 32

