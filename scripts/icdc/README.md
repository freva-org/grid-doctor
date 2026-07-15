## IMERG

### Initializing the dataset:

In case it's the first time loadding the data one need more resources:
 - ```bash
    sbatch -p shared -Ak20200 --mem 100G --time 3-00:00:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py imerg init /work/ks1387/gw/data/icdc/healpix/atmosphere/IMERG/PT30M/')
    `
 - Afterwards:
    ```bash
    python3 scripts/icdc/main.py imerg init
    ```
But note that ~450K files will be loaded as a single datas. The job took about 41 hours, and memory usage peaked at 65GiB

### Writing
 - Sequentially:
    ```bash
    python3 scripts/icdc/main.py write
    ```

 - In parallel using array jobs:
    ```
    sbatch -p compute -Ak20200 --mem 16G --array=0-999 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py imerg write --files-per-batch=443  /work/ks1387/gw/data/icdc/healpix/atmosphere/IMERG/PT30M/')
    ```
    Given the time axis is 442669 long, the batch size was set to 443 because one can only submit a maximum of 1000 jobs using the array option. 
    Each invidual job is taking roughly 10 minutes to write its region (consisting of 443 timesteps)


## MODIS ATM

### Initialize
```bash
sbatch -p shared -Ak20200 --mem 100G --time 3-00:00:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py modis-atm-aqua init /work/ks1387/gw/data/icdc/healpix/atmosphere/MODIS/aqua/P1D/')
```

### Write
```
sbatch -p compute -Ak20200 --mem 16G --array=0-542 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py modis-atm-aqua write --files-per-batch=16 /work/ks1387/gw/data/icdc/healpix/atmosphere/MODIS/aqua/P1D/')
```


## CERES ATM

### Initialize:

These are HDF4 files with 24 timesteps per file that need to be preprocessed before merging. This takes a lot of time!
```
sbatch -p shared -Ak20200 --mem 100G --time=5-00:00:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3  scripts/icdc/main.py ceres init /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
```

### Write

The approach here had to change, writing batches of files per jobs to reduce the amount of jobs needed fails, even after 6 hours the writting of 223 files in a single job (9252 files * 24 timesteps / 1000 jobs + 1) to the respective region didn't finish.

The idea now is that we launch a job per file meaning that we need to submit 9252 jobs, but array jobs are limited to 1000 at a time so we need to split this into individual submissions:
```
sbatch -p compute -Ak20200 --mem 16G --array=0-999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=1000-1999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=2000-2999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=3000-3999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=4000-4999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=5000-5999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=6000-6999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=7000-7999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=8000-8999 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
sbatch -p compute -Ak20200 --mem 16G --array=9000-9252 --time=0-00:03:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch 1 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
```
