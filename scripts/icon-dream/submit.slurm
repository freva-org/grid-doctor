#!/bin/bash
#SBATCH -J icon-dream2healpix
#SBATCH -p compute
#SBATCH -n 12
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH -t 08:00:00
#SBATCH --signal=15@10
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail
SCRIPT="$SLURM_SUBMIT_DIR/convert.py"

srun python "$SCRIPT" "$@"
