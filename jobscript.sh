#!/usr/bin/env bash
#SBATCH --job-name TensorTesting(fashion_MNIST)
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --mail-type=ALL # NONE, BEGIN, END, FAIL, REQUEUE, ALL TIME_LIMIT, TIME_LIMIT_90, etc
#SBATCH --mail-user=ksla@create.aau.dk
#SBATCH --dependency=aftercorr:498 # More info slurm head node: `man --pager='less -p \--dependency' sba$
#SBATCH --gres=gpu:1
#SBATCH --qos=normal # possible values: short, normal

srun echo 'Running test_sandbox'
srun singularity run --nv test.sif

srun echo 'all done'