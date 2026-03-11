#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_slurm_pipeline.sh [options]

Required:
  --script-path PATH            Path to convert_pipeline.py
  --work-dir PATH               Shared pipeline work directory
  --s3-bucket NAME              Target S3 bucket for final pack stage

Optional pipeline args:
  --freq VALUE                  hourly|daily|monthly|fx (default: hourly)
  --variable NAME               Variable to process; repeatable (default: t_2m tot_prec)
  --time-start VALUE            Start time (default: 2010-01-01T00:00)
  --time-end VALUE              End time (default: now)
  --download-workers N          Parallel downloads in prepare stage (default: 8)
  --max-array-parallel N        Max concurrent worker array tasks (default: 8)
  --local-tmp-base PATH         Base dir for node-local worker tmp (default: /tmp/$USER/icon-dream)
  --manifest PATH               Override manifest path (default: <work-dir>/manifests/job_manifest.json)
  --s3-endpoint URL             Optional S3 endpoint passed to pack stage
  --s3-credentials-file PATH    Optional S3 credentials file passed to pack stage
  --extra-arg ARG               Extra arg forwarded to convert_pipeline.py; repeatable

Optional SBATCH routing:
  --account NAME                SLURM account (default: ch1187)
  --partition NAME              SLURM partition (default: compute)
  --qos NAME                    Optional QOS
  --constraint VALUE            Optional constraint
  --reservation NAME            Optional reservation

Optional prepare resources:
  --prepare-time HH:MM:SS       Default: 04:00:00
  --prepare-cpus N              Default: 4
  --prepare-mem VALUE           Default: 16G

Optional worker resources:
  --worker-time HH:MM:SS        Default: 12:00:00
  --worker-cpus N               Default: 8
  --worker-mem VALUE            Default: 64G

Optional pack resources:
  --pack-time HH:MM:SS          Default: 12:00:00
  --pack-cpus N                 Default: 4
  --pack-mem VALUE              Default: 32G

Other:
  --job-name-prefix NAME        Prefix for SLURM job names (default: icon)
  --help                        Show this help

Example:
  ./submit_slurm_pipeline.sh \
    --script-path /path/to/convert_pipeline.py \
    --work-dir /scratch/k/$USER/icon-dream-run \
    --s3-bucket icon-dream \
    --variable t_2m \
    --variable tot_prec \
    --freq hourly \
    --time-start 2010-01-01T00:00 \
    --time-end now \
    --account ch1187 \
    --partition compute
EOF
}

# ----------------------------
# Defaults
# ----------------------------

ACCOUNT="ch1187"
PARTITION="compute"
QOS=""
CONSTRAINT=""
RESERVATION=""

JOB_NAME_PREFIX="icon"

PREPARE_TIME="04:00:00"
PREPARE_CPUS="4"
PREPARE_MEM="16G"

WORKER_TIME="12:00:00"
WORKER_CPUS="8"
WORKER_MEM="64G"

PACK_TIME="12:00:00"
PACK_CPUS="4"
PACK_MEM="32G"

FREQ="hourly"
TIME_START="2010-01-01T00:00"
TIME_END="now"
DOWNLOAD_WORKERS="8"
MAX_ARRAY_PARALLEL="8"

PYTHON_BIN="python"
SCRIPT_PATH=""
WORK_DIR=""
MANIFEST=""
S3_BUCKET=""
S3_ENDPOINT=""
S3_CREDENTIALS_FILE=""
LOCAL_TMP_BASE="/tmp/${USER}/icon-dream"

declare -a VARIABLES=()
declare -a EXTRA_ARGS=()

# ----------------------------
# Parse args
# ----------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --script-path)
      SCRIPT_PATH="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --s3-bucket)
      S3_BUCKET="$2"
      shift 2
      ;;
    --s3-endpoint)
      S3_ENDPOINT="$2"
      shift 2
      ;;
    --s3-credentials-file)
      S3_CREDENTIALS_FILE="$2"
      shift 2
      ;;
    --freq)
      FREQ="$2"
      shift 2
      ;;
    --variable)
      VARIABLES+=("$2")
      shift 2
      ;;
    --time-start)
      TIME_START="$2"
      shift 2
      ;;
    --time-end)
      TIME_END="$2"
      shift 2
      ;;
    --download-workers)
      DOWNLOAD_WORKERS="$2"
      shift 2
      ;;
    --max-array-parallel)
      MAX_ARRAY_PARALLEL="$2"
      shift 2
      ;;
    --local-tmp-base)
      LOCAL_TMP_BASE="$2"
      shift 2
      ;;
    --extra-arg)
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --constraint)
      CONSTRAINT="$2"
      shift 2
      ;;
    --reservation)
      RESERVATION="$2"
      shift 2
      ;;
    --prepare-time)
      PREPARE_TIME="$2"
      shift 2
      ;;
    --prepare-cpus)
      PREPARE_CPUS="$2"
      shift 2
      ;;
    --prepare-mem)
      PREPARE_MEM="$2"
      shift 2
      ;;
    --worker-time)
      WORKER_TIME="$2"
      shift 2
      ;;
    --worker-cpus)
      WORKER_CPUS="$2"
      shift 2
      ;;
    --worker-mem)
      WORKER_MEM="$2"
      shift 2
      ;;
    --pack-time)
      PACK_TIME="$2"
      shift 2
      ;;
    --pack-cpus)
      PACK_CPUS="$2"
      shift 2
      ;;
    --pack-mem)
      PACK_MEM="$2"
      shift 2
      ;;
    --job-name-prefix)
      JOB_NAME_PREFIX="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# ----------------------------
# Validate args
# ----------------------------

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "Error: --script-path is required" >&2
  exit 2
fi

if [[ -z "${WORK_DIR}" ]]; then
  echo "Error: --work-dir is required" >&2
  exit 2
fi

if [[ -z "${S3_BUCKET}" ]]; then
  echo "Error: --s3-bucket is required" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Error: script not found: ${SCRIPT_PATH}" >&2
  exit 2
fi

if [[ ${#VARIABLES[@]} -eq 0 ]]; then
  VARIABLES=("t_2m" "tot_prec")
fi

if [[ -z "${MANIFEST}" ]]; then
  MANIFEST="${WORK_DIR}/manifests/job_manifest.json"
fi

mkdir -p "${WORK_DIR}/logs"

# ----------------------------
# Helper builders
# ----------------------------

build_optional_sbatch_args() {
  local -a args=()
  args+=(--account="${ACCOUNT}")
  args+=(--partition="${PARTITION}")

  if [[ -n "${QOS}" ]]; then
    args+=(--qos="${QOS}")
  fi
  if [[ -n "${CONSTRAINT}" ]]; then
    args+=(--constraint="${CONSTRAINT}")
  fi
  if [[ -n "${RESERVATION}" ]]; then
    args+=(--reservation="${RESERVATION}")
  fi

  printf '%s\n' "${args[@]}"
}

build_python_common_args() {
  local -a args=()
  args+=(--work-dir "${WORK_DIR}")
  args+=(--manifest "${MANIFEST}")
  args+=(--freq "${FREQ}")
  args+=(--time "${TIME_START}" "${TIME_END}")

  for var in "${VARIABLES[@]}"; do
    args+=(--variables "${var}")
  done

  if [[ -n "${S3_BUCKET}" ]]; then
    args+=(--s3-bucket "${S3_BUCKET}")
  fi
  if [[ -n "${S3_ENDPOINT}" ]]; then
    args+=(--s3-endpoint "${S3_ENDPOINT}")
  fi
  if [[ -n "${S3_CREDENTIALS_FILE}" ]]; then
    args+=(--s3-credentials-file "${S3_CREDENTIALS_FILE}")
  fi
  for extra in "${EXTRA_ARGS[@]}"; do
    args+=("${extra}")
  done

  printf '%q ' "${args[@]}"
}

COMMON_PY_ARGS="$(build_python_common_args)"
mapfile -t OPTIONAL_SBATCH_ARGS < <(build_optional_sbatch_args)

# ----------------------------
# Submit prepare job
# ----------------------------

prepare_jobid="$(
  sbatch --parsable \
    "${OPTIONAL_SBATCH_ARGS[@]}" \
    --job-name="${JOB_NAME_PREFIX}-prepare" \
    --output="${WORK_DIR}/logs/prepare-%j.out" \
    --error="${WORK_DIR}/logs/prepare-%j.err" \
    --time="${PREPARE_TIME}" \
    --cpus-per-task="${PREPARE_CPUS}" \
    --mem="${PREPARE_MEM}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${WORK_DIR}/logs"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --mode prepare \
  ${COMMON_PY_ARGS} \
  --download-workers "${DOWNLOAD_WORKERS}"
EOF
)"

echo "prepare job: ${prepare_jobid}"

# ----------------------------
# Submit follow-up orchestration job
# ----------------------------

submit_jobid="$(
  sbatch --parsable \
    "${OPTIONAL_SBATCH_ARGS[@]}" \
    --dependency="afterok:${prepare_jobid}" \
    --job-name="${JOB_NAME_PREFIX}-submit" \
    --output="${WORK_DIR}/logs/submit-%j.out" \
    --error="${WORK_DIR}/logs/submit-%j.err" \
    --time="00:20:00" \
    --cpus-per-task="1" \
    --mem="1G" <<EOF
#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${WORK_DIR}/logs"

manifest="${MANIFEST}"
if [[ ! -f "\${manifest}" ]]; then
  echo "Manifest not found: \${manifest}" >&2
  exit 1
fi

count=\$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
data = json.loads(Path("${MANIFEST}").read_text())
print(len(data["jobs"]))
PY
)

if [[ "\${count}" -le 0 ]]; then
  echo "No jobs found in manifest: \${manifest}" >&2
  exit 1
fi

last_index=\$((count - 1))

worker_jobid=\$(
  sbatch --parsable \
    ${OPTIONAL_SBATCH_ARGS[*]} \
    --job-name="${JOB_NAME_PREFIX}-worker" \
    --output="${WORK_DIR}/logs/worker-%A_%a.out" \
    --error="${WORK_DIR}/logs/worker-%A_%a.err" \
    --time="${WORKER_TIME}" \
    --cpus-per-task="${WORKER_CPUS}" \
    --mem="${WORKER_MEM}" \
    --array="0-\${last_index}%${MAX_ARRAY_PARALLEL}" <<WORKER_EOF
#!/usr/bin/env bash
set -euo pipefail

export TMPDIR="${LOCAL_TMP_BASE}/\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TMPDIR}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --mode worker \
  ${COMMON_PY_ARGS} \
  --local-tmp-dir "\${TMPDIR}"
WORKER_EOF
)

echo "worker array: \${worker_jobid}"

pack_jobid=\$(
  sbatch --parsable \
    ${OPTIONAL_SBATCH_ARGS[*]} \
    --dependency="afterok:\${worker_jobid}" \
    --job-name="${JOB_NAME_PREFIX}-pack" \
    --output="${WORK_DIR}/logs/pack-%j.out" \
    --error="${WORK_DIR}/logs/pack-%j.err" \
    --time="${PACK_TIME}" \
    --cpus-per-task="${PACK_CPUS}" \
    --mem="${PACK_MEM}" <<PACK_EOF
#!/usr/bin/env bash
set -euo pipefail

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --mode pack \
  ${COMMON_PY_ARGS}
PACK_EOF
)

echo "pack job: \${pack_jobid}"
EOF
)"

echo "submit-followup job: ${submit_jobid}"
echo "pipeline submitted"
