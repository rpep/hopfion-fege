#!/bin/bash

#SBATCH --ntasks-per-node=5     # Tasks per node
#SBATCH --nodes=1               # Number of nodes requested
#SBATCH --time 02:00:00         # Time requestsed (max 48 hours)   
#SBATCH --mail-type=begin       # E-mails you when it starts
#SBATCH --mail-type=end         # E-mails it you when it ends


module load singularity
mkdir -p /tmp/singularity/mnt/container/ 
mkdir -p /tmp/singularity/mnt/final/
mkdir -p /tmp/singularity/mnt/overlay/
mkdir -p /tmp/singularity/mnt/session/
mkdir -p /tmp/finmag-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
cp -r ../finmag/* /tmp/finmag-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}

# Run Python script in the singularity container.
singularity exec -B /mainfs/scratch/${USER}:/mainfs/scratch/${USER} finmag.img python ../src/script.py



