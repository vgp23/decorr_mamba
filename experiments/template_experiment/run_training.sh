#!/bin/bash



source /opt/anaconda3/etc/profile.d/conda.sh
conda activate thesis_env

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

python train.py > "logs/output_$timestamp.log" 2>&1

conda deactivate

