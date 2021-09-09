#!/bin/bash
#SBATCH --partition=iris --qos=normal
#SBATCH --exclude=iris3
#SBATCH --include=iris1,iris2,iris4
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#
# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#
#SBATCH --job-name="lifelong_learning_pets_reacher"
#SBATCH --output=lifelong_learning_pets_reacher-%j.out
#
# only use the following if you want email notification
#SBATCH --mail-user=maxsobolmark@stanford.edu
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
pecho "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# can try the following to list out which GPU you have access to
srun echo "Running from IRIS"
srun echo "Experiment: Reacher state PETS"
srun /iris/u/maxsobolmark/pytorch/bin/python3 -m mbrl.examples.main_lifelong_learning algorithm=fsrl overrides=pets_reacher_lifelong_learning

# done
echo "Done"
