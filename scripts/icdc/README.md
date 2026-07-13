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

```
sbatch -p shared -Ak20200 --mem 100G --time=3-00:00:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3  scripts/icdc/main.py ceres init /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/')
```

### Write

9252 files each with 24 timesteps batched into 1000 jobs leads to 223 files per batch  `echo '24 * 9252 / 1000 + 1' | bc`

```
sbatch -p compute -Ak20200 --mem 16G --array=0-999 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py ceres write --files-per-batch=223 /work/ks1387/gw/data/icdc/healpix/atmosphere/CERES/PT1H/
```
