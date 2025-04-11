#!/bin/bash
#SBATCH --job-name=ego4d_0
#SBATCH --time=14:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --cpus-per-task=8
#SBATCH --error=/home/modelrep/schaumloeffel/solo-learn/run_logs/log_%j.err
#SBATCH --output=/home/modelrep/schaumloeffel/solo-learn/run_logs/log_%j.out

eval "$(conda shell.bash hook)"
conda activate ml
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/ml/lib

#export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd $HOME/solo-learn

srun python run_eval_convnext.py --env hlr
