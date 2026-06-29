## IMERG

### Initializing the dataset:

In case it's the first time loadding the data one need more resources:
 - ```bash
    sbatch -p shared -Ak20200 --mem 100G --time 3-00:00:00 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py init /work/ks1387/gw/data/icdc/healpix/atmosphere/IMERG/PT30M/')
    ```
 - Afterwards:
    ```bash
    python3 scripts/icdc/main.py init
    ```
But note that ~450K files will be loaded as a single datas. The job took about 41 hours, and memory usage peaked at 65GiB

### Writing
 - Sequentially:
    ```bash
    python3 scripts/icdc/main.py write
    ```

 - In parallel using array jobs:
    ```
    sbatch -p compute -Ak20200 --mem 16G --array=0-100 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py write --batch-size=128  /work/ks1387/gw/data/icdc/healpix/atmosphere/IMERG/PT30M/')
    ```
