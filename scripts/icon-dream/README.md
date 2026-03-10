# ICON-DREAM

## What is icon-dream?

Icon dream is a reanalysis product from the German Weather Service.


## How to run:

1. Create and download your s3-secrets file from [https://eu-dkrz-3.dkrz.cloud/access-keys](https://eu-dkrz-3.dkrz.cloud/access-keys)
2. Put the secrets files somewhere into your home on levante.
3. Install the requirements
```console
python -m pip install -r requirements.txt
```
4. Use the [submit.slurm](submit.slurm) script to submit a convert job. You will
   need need to fine tune the convert script. Here are the following options:
```console
python convert.py --helpl
sage: convert.py [-h] [--s3-endpoint S3_ENDPOINT] [--s3-credentials-file S3_CREDENTIALS_FILE] [-v]
                  [--variables [{aswdifd_s,aswdir_s,clct,den,p,pmsl,ps,qv,qv_s,t,td_2m,tke,tmax_2m,tmin_2m,tot_prec,t_2m,u,u_10m,v,vmax_10m,v_10m,ws,ws_10m,z0} ...]]
                  [--freq {hourly,daily,monthly,fx}] [--time TIME TIME]
                  s3-bucket

Download an convert ICON-DREAM data.

Positional Arguments:
  s3-bucket             S3 target bucket.

Options:
  -h, --help            show this help message and exit
  --s3-endpoint S3_ENDPOINT
                        S3 endpoint URL. (default: https://s3.eu-dkrz-3.dkrz.cloud)
  --s3-credentials-file S3_CREDENTIALS_FILE
                        Path to a JSON file with accessKey/secretKey. (default: /home/k/k204230/.s3-credentials.json)
  -v, --verbose         Increase verbosity (repeat for more: -v, -vv, -vvv). (default: 0)
  --variables [{aswdifd_s,aswdir_s,clct,den,p,pmsl,ps,qv,qv_s,t,td_2m,tke,tmax_2m,tmin_2m,tot_prec,t_2m,u,u_10m,v,vmax_10m,v_10m,ws,ws_10m,z0} ...]
  --freq, --time-frequency {hourly,daily,monthly,fx}
                        Time frequency of the data. (default: hourly)
  --time TIME TIME
```
To submit the job choose your slurm partition and add any arguments for the
`convert.py` script for example:

```console
sbatch -A ch1187 submit.slurm icon-dream --freq fx
```


> [!IMPORTANT]
> Once you've downloaded the s3 secrets file apply `chmod 600` to it:
> `chmod 600 ~/.s3-credentials.json`
