#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=20G
#SBATCH --partition=rosa.p
#SBATCH --time=0-4:00
#SBATCH --output=slurm_reports/slurm.%j.out
#SBATCH --error=slurm_reports/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bjarne.schultze1@uol.de

module load hpc-env/13.1
module load Anaconda3

conda activate /user/seck8229/.conda/envs/spiking_neural_network_model

# Run simulation with spiking neuronal network model
python sknnm_sequential_simulation.py