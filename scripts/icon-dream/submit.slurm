#!/bin/bash
#SBATCH -J icon-dream2healpix
#SBATCH -p compute
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH --signal=15@10
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail
SCRIPT="$SLURM_SUBMIT_DIR/convert.py"

srun python "$SCRIPT" "$@"
