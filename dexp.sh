#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=DDRtrain
#SBATCH --output=slurm/DDRtrain.out
#SBATCH --error=slurm/DDRtrain.err
#SBATCH --account=nn10058k
#SBATCH --partition=accel ## Selected partition
#SBATCH --mem-per-cpu=24000 ## Memory allocated to each task
#SBATCH --ntasks=1 ## Number of tasks that will be allocated
#SBATCH --gpus=2 ## GPUs allocated per task

nvidia-smi
echo "This script is running on "
echo "-------------------------"
which python
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23_slim.yaml 

echo "-------------------------"

echo $HOSTNAME

echo "-------------------------"