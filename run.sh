#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH -C gmem80
#SBATCH -c 12
#SBATCH --mem 64G
#SBATCH --job-name pmc_pretrain
#SBATCH --output output.log
#SBATCH --error error.log

python scripts/model/03_train_projectors.py \
  projector_train.resume_projector_path=outputs/train/projectors/pmc_oa_caption/last/projector.pt \
  projector_train.num_epochs=2
