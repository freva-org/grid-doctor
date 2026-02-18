#! /usr/bin/bash

set -e

function makewrapper {
cat << EOF
#! /bin/bash
#SBATCH --mem=100GB
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00

srun python3 scripts/era5/convert.py --slice-size=\$1
EOF
}

function run_init {
    echo srun --account=$(groups|cut -d' ' -f1) -p shared --mem=100GB --time=1-00:00:00 python3 scripts/era5/convert.py --init 
}


function run_update {
    local time_end=$(curl -sL https://s3.eu-dkrz-1.dkrz.cloud/icdc/healpix/era5.zarr/time/.zarray | jq '.shape[]')
    local slicesize=$(( 48 * 30 ))
    local tasks=$((time_end / slicesize))
    #tasks=1
    makewrapper > era5wrapper.sh
    echo "sbatch --account=$(groups|cut -d' ' -f1) -p shared --array=0-${tasks} era5wrapper.sh ${slicesize}"
}


run_init
run_update

