#!/bin/bash
#SBATCH --job-name=entity_matching
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gres=gpu:2
#SBATCH -n 1

# load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

python main.py