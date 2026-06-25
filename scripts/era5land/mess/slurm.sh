#! /usr/bin/bash

set -euo pipefail

CONFIG_PATH="${ERA5LAND_CONFIG_PATH:-scripts/era5land/config.json}"
SLICE_SIZE="${ERA5LAND_SLICE_SIZE:-1440}"

function makewrapper {
cat << EOF
#! /bin/bash
#SBATCH --mem=100GB
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00

srun python3 scripts/era5land/convert.py --config=${CONFIG_PATH} --slice-size=\$1
EOF
}

function run_init {
    echo srun --account=$(groups|cut -d' ' -f1) -p shared --mem=100GB --time=1-00:00:00 python3 scripts/era5land/convert.py --config=${CONFIG_PATH} --init
}


function run_update {
    echo "# Adjust ERA5LAND_STORE_TIME_LENGTH or inspect the destination store before scheduling updates."
    local time_end=${ERA5LAND_STORE_TIME_LENGTH:-0}
    local tasks=$((time_end / SLICE_SIZE))
    makewrapper > era5land-wrapper.sh
    echo "sbatch --account=$(groups|cut -d' ' -f1) -p shared --array=0-${tasks} era5land-wrapper.sh ${SLICE_SIZE}"
}


run_init
run_update
