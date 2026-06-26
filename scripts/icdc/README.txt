## IMERG

```bash
python3 scripts/icdc/main.py init
```
```bash
python3 scripts/icdc/main.py write
```

or


```bash
sbatch -p compute -Ak20200 --mem 16G --array=0-100 <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/icdc/main.py write --batch-size=128  /work/ks1387/gw/data/icdc/healpix/atmosphere/IMERG/PT30M/')
```
