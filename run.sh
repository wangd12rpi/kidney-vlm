#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH -C gmem80
#SBATCH -c 12
#SBATCH --mem 80G
#SBATCH --job-name pmc_pretrain
#SBATCH --output output.log
#SBATCH --error error.log


python scripts/model/03_train_projectors.py