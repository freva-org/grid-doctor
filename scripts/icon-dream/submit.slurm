#!/usr/bin/env bash
#SBATCH -J icon-dream2healpix
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH --signal=15@10
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}/submit.slurm}"
CONVERT_PY="${CONVERT_PY:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}/convert.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PARTITION="${PARTITION:-compute}"
ACCOUNT="${ACCOUNT:-}"
MEMORY="${MEMORY:-0}"
WORKER_ARRAY_PARALLELISM="${WORKER_ARRAY_PARALLELISM:-16}"

PLAN_CPUS="${PLAN_CPUS:-2}"
PREPARE_CPUS="${PREPARE_CPUS:-16}"
WORKER_CPUS="${WORKER_CPUS:-32}"
FINALIZE_CPUS="${FINALIZE_CPUS:-8}"
DISPATCH_CPUS="${DISPATCH_CPUS:-1}"

PLAN_TIME="${PLAN_TIME:-00:20:00}"
PREPARE_TIME="${PREPARE_TIME:-08:00:00}"
WORKER_TIME="${WORKER_TIME:-08:00:00}"
FINALIZE_TIME="${FINALIZE_TIME:-08:00:00}"
DISPATCH_TIME="${DISPATCH_TIME:-00:10:00}"

SBATCH_ACCOUNT_ARGS=()
if [[ -n "$ACCOUNT" ]]; then
    SBATCH_ACCOUNT_ARGS=(-A "$ACCOUNT")
fi

export SUBMIT_SCRIPT CONVERT_PY PYTHON_BIN
export PARTITION ACCOUNT MEMORY WORKER_ARRAY_PARALLELISM
export PLAN_CPUS PREPARE_CPUS WORKER_CPUS FINALIZE_CPUS DISPATCH_CPUS
export PLAN_TIME PREPARE_TIME WORKER_TIME FINALIZE_TIME DISPATCH_TIME

usage() {
    cat <<'USAGE'
Usage:
  ./submit.slurm submit <s3_bucket> [convert.py options ...]
  sbatch submit.slurm stage <plan|prepare|worker|finalize|dispatch> <s3_bucket> [convert.py options ...]

Examples:
  ACCOUNT=bb1234 ./submit.slurm submit my-bucket \
    --freq hourly \
    --variables t_2m tot_prec \
    --time 2025-01-01T00:00:00Z now \
    --update-only

Environment overrides:
  ACCOUNT=bb1234
  PARTITION=compute
  PLAN_CPUS=2 PREPARE_CPUS=16 WORKER_CPUS=32 FINALIZE_CPUS=8
  PLAN_TIME=00:20:00 PREPARE_TIME=08:00:00 WORKER_TIME=08:00:00 FINALIZE_TIME=08:00:00
  WORKER_ARRAY_PARALLELISM=16
  PYTHON_BIN=python
USAGE
}

extract_run_dir() {
    local previous=""
    local arg
    for arg in "$@"; do
        if [[ "$previous" == "--run-dir" ]]; then
            printf '%s\n' "$arg"
            return 0
        fi
        case "$arg" in
            --run-dir=*)
                printf '%s\n' "${arg#--run-dir=}"
                return 0
                ;;
        esac
        previous="$arg"
    done
    return 1
}

submit_dag() {
    if [[ ! -f "$CONVERT_PY" ]]; then
        echo "convert.py not found: $CONVERT_PY" >&2
        exit 1
    fi

    local common_args=("$@")
    local run_dir
    local u_number
    u_number="${USER:0:1}"
    SCRATCH="/scratch/${u_number}/${USER}"
    if ! run_dir="$(extract_run_dir "${common_args[@]}")"; then
        run_dir="${RUN_DIR:-${SCRATCH}/grid-doctor/icon-dream-run-$(date -u +%Y%m%dT%H%M%SZ)}"
        common_args+=(--run-dir "$run_dir")
    fi
    local plan_id
    local prepare_id
    local dispatch_id
    local logs_dir
    logs_dir="$run_dir/logs"
    mkdir -p "$run_dir/logs"

    plan_id="$({
        sbatch --parsable \
            "${SBATCH_ACCOUNT_ARGS[@]}" \
            -J icon-dream-plan \
            -p "$PARTITION" \
            -N 1 \
            --ntasks=1 \
            -c "$PLAN_CPUS" \
            --mem="$MEMORY" \
            -t "$PLAN_TIME" \
            --output="$logs_dir/plan-%j.out" \
            --error="$logs_dir/plan-%j.err" \
            --export=ALL \
            "$SUBMIT_SCRIPT" stage plan "${common_args[@]}"
    } | tail -n 1)"

    prepare_id="$({
        sbatch --parsable \
            "${SBATCH_ACCOUNT_ARGS[@]}" \
            -J icon-dream-prepare \
            -p "$PARTITION" \
            -N 1 \
            --ntasks=1 \
            -c "$PREPARE_CPUS" \
            --mem="$MEMORY" \
            -t "$PREPARE_TIME" \
            --output="$logs_dir/prepare-%j.out" \
            --error="$logs_dir/prepare-%j.err" \
            --dependency="afterok:${plan_id}" \
            --export=ALL \
            "$SUBMIT_SCRIPT" stage prepare "${common_args[@]}"
    } | tail -n 1)"

    dispatch_id="$({
        sbatch --parsable \
            "${SBATCH_ACCOUNT_ARGS[@]}" \
            -J icon-dream-dispatch \
            -p "$PARTITION" \
            -N 1 \
            --ntasks=1 \
            -c "$DISPATCH_CPUS" \
            --mem="$MEMORY" \
            -t "$DISPATCH_TIME" \
            --output="$logs_dir/dispatch-%j.out" \
            --error="$logs_dir/dispatch-%j.err" \
            --dependency="afterok:${prepare_id}" \
            --export=ALL \
            "$SUBMIT_SCRIPT" stage dispatch "${common_args[@]}"
    } | tail -n 1)"

    cat <<EOF
Submitted ICON-DREAM DAG
  run_dir : $run_dir
  account : ${ACCOUNT:-<default>}
  plan    : $plan_id
  prepare : $prepare_id
  dispatch: $dispatch_id
EOF
}

dispatch_followup() {
    local common_args=("$@")
    local run_dir
    local log_dir
    run_dir="$(extract_run_dir "${common_args[@]}")"
    local manifest_path="$run_dir/manifest.json"

    if [[ ! -f "$manifest_path" ]]; then
        echo "Manifest not found: $manifest_path" >&2
        exit 1
    fi
    log_dir="${run_dir}/logs"
    mkdir -p "${log_dir}"
    local item_count
    item_count="$($PYTHON_BIN - "$manifest_path" <<'PY'
import json
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    payload = json.load(handle)
print(len(payload.get('source_items', [])))
PY
)"
    item_count="${item_count//[[:space:]]/}"

    local finalize_id
    if [[ "$item_count" == "0" ]]; then
        finalize_id="$({
            sbatch --parsable \
                "${SBATCH_ACCOUNT_ARGS[@]}" \
                -J icon-dream-finalize \
                -p "$PARTITION" \
                -N 1 \
                --ntasks=1 \
                -c "$FINALIZE_CPUS" \
                --mem="$MEMORY" \
                -t "$FINALIZE_TIME" \
                --output="$log_dir/dispatch-%j.out" \
                --error="$log_dir/dispatch-%j.err" \
                --export=ALL \
                "$SUBMIT_SCRIPT" stage finalize "${common_args[@]}"
        } | tail -n 1)"

        cat <<EOF
Manifest contains 0 items.
Submitted finalize only:
  finalize: $finalize_id
EOF
        return 0
    fi

    local worker_max_index=$((item_count - 1))
    local worker_id
    worker_id="$({
        sbatch --parsable \
            "${SBATCH_ACCOUNT_ARGS[@]}" \
            -J icon-dream-worker \
            -p "$PARTITION" \
            -N 1 \
            --ntasks=1 \
            -c "$WORKER_CPUS" \
            --mem="$MEMORY" \
            -t "$WORKER_TIME" \
            --output="$log_dir/worker-%j.out" \
            --error="$log_dir/worker-%j.err" \
            --array="0-${worker_max_index}%${WORKER_ARRAY_PARALLELISM}" \
            --export=ALL \
            "$SUBMIT_SCRIPT" stage worker "${common_args[@]}"
    } | tail -n 1)"

    finalize_id="$({
        sbatch --parsable \
            "${SBATCH_ACCOUNT_ARGS[@]}" \
            -J icon-dream-finalize \
            -p "$PARTITION" \
            -N 1 \
            --ntasks=1 \
            -c "$FINALIZE_CPUS" \
            --mem="$MEMORY" \
            -t "$FINALIZE_TIME" \
            --output="$log_dir/finalize-%j.out" \
            --error="$log_dir/finalize-%j.err" \
            --dependency="afterok:${worker_id}" \
            --export=ALL \
            "$SUBMIT_SCRIPT" stage finalize "${common_args[@]}"
    } | tail -n 1)"

    cat <<EOF
Manifest contains $item_count items.
Submitted follow-up jobs
  worker array: $worker_id (0-${worker_max_index}%${WORKER_ARRAY_PARALLELISM})
  finalize    : $finalize_id
EOF
}

run_stage() {
    local stage="$1"
    shift

    case "$stage" in
        plan|prepare|worker|finalize)
            exec "$PYTHON_BIN" "$CONVERT_PY" "$stage" "$@"
            ;;
        dispatch)
            dispatch_followup "$@"
            ;;
        *)
            echo "Unknown stage: $stage" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main() {
    if [[ "$#" -lt 1 ]]; then
        usage >&2
        exit 2
    fi

    local command="$1"
    shift

    case "$command" in
        submit)
            submit_dag "$@"
            ;;
        stage)
            if [[ "$#" -lt 1 ]]; then
                echo "Missing stage name." >&2
                exit 2
            fi
            run_stage "$@"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "Unknown command: $command" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
