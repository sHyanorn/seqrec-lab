#!/usr/bin/env bash
# Steps of the batch workflow (.github/workflows/batch.yml). Inputs arrive as environment
# variables and are checked here before anything uses them.
#   watchdog-running | check-inputs | machine | status STATE TEXT | plan | profile |
#   data-unchanged | export | results |
#   commit MESSAGE PATH...
set -euo pipefail

DATASETS="ml-1m ml-20m steam video-games"

die() {
    echo "$*" >&2
    exit 1
}

# The batch a plan file belongs to; the plan must be named after it.
plan_batch() {
    [[ "${PLAN:-}" =~ ^[A-Za-z0-9_-]+\.toml$ ]] || die "plan: not a plan file name"
    local file="configs/plans/$PLAN" batch=${PLAN%.toml}
    [ -f "$file" ] || die "plan: $file does not exist"
    grep -qxF "batch = \"$batch\"" "$file" || die "plan: the batch in $PLAN is not named $batch"
    echo "$batch"
}

# PROFILE_ARGS as profile_step.py arguments, one per line; only the forms listed here pass.
profile_args() {
    local -a words
    local dataset="" arm="" i=0 word value
    read -ra words <<< "${PROFILE_ARGS:-}"
    while [ "$i" -lt "${#words[@]}" ]; do
        word=${words[$i]} value=${words[$((i + 1))]:-}
        case "$word" in
            --dataset)
                [[ " $DATASETS " == *" $value "* ]] || die "profile_args: unknown dataset"
                dataset=$value i=$((i + 1))
                ;;
            --arm)
                [[ "$value" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || die "profile_args: not an arm name"
                arm=$value i=$((i + 1))
                ;;
            --steps | --warmup)
                [[ "$value" =~ ^[0-9]{1,6}$ ]] || die "profile_args: $word needs a number"
                i=$((i + 1))
                ;;
            --fixed-shapes | --fused-adam | --cuda-graph | --sync-debug | --check) ;;
            *) die "profile_args: unexpected argument" ;;
        esac
        i=$((i + 1))
    done
    [ -n "$dataset" ] && [ -n "$arm" ] || die "profile_args: --dataset and --arm are required"
    printf '%s\n' "${words[@]}"
}

check_inputs() {
    [ "${GITHUB_REF:-}" = refs/heads/main ] || die "ref: batches run from main only"
    case "${TASK:-}" in
        setup) ;;
        test | export) plan_batch > /dev/null ;;
        run | tune)
            plan_batch > /dev/null
            [[ "${HOURS:-}" =~ ^([1-9]|1[0-9]|2[0-2])$ ]] || die "hours: a whole number from 1 to 22"
            ;;
        profile)
            profile_args > /dev/null
            [[ "${REPEATS:-}" =~ ^([1-9]|1[0-9]|20)$ ]] || die "repeats: a whole number from 1 to 20"
            ;;
        *) die "task: unknown" ;;
    esac
}

# What the job runs on, for the log. The runner is registered on a machine that may have no
# GPU, so the driver is checked here, by every job.
machine() {
    local driver
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    [ "${driver%%.*}" -ge 580 ] || die "driver branch below 580: the CUDA 13 wheels will not run here"
    lscpu | grep "Model name"
    nproc
    uname -r
    uv --version
    echo "runner $("${RUNNER_DIR:-/workspace/actions-runner}/bin/Runner.Listener" --version)"
    uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.__version__, torch.cuda.get_device_name(0))"
    uv run --no-sync python -c "import torch; from seqrec.runner.run import cpu_allocation, host_signature; print('host', host_signature(torch.device('cuda'))); print('cpus', *cpu_allocation())"
    # The signature does not use the platform's count; it is printed for comparison.
    echo "cgroup filesystem $(stat -fc %T /sys/fs/cgroup 2>&1), RUNPOD_CPU_COUNT ${RUNPOD_CPU_COUNT:-unset}"
    tail -n 20 "$SEQREC_HOME/runner.log" # when the watchdog started the runner and when it listened
}

# Commit status "batch/<run id>" on the commit the job runs from. The job log is readable
# only after the job ends; this is readable while it runs. Never fails the step.
status() {
    local text
    text=$(printf '%s' "$2" | tr -cd 'A-Za-z0-9 ._:,/-' | cut -c1-140)
    curl -fsS -m 20 -o /dev/null -H @- \
        "$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/statuses/$GITHUB_SHA" \
        -d "{\"state\":\"$1\",\"context\":\"batch/$GITHUB_RUN_ID\",\"description\":\"$text\",\"target_url\":\"$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID\"}" \
        <<< "Authorization: Bearer $GH_TOKEN" || true
}

plan() {
    local batch runs code=0 ticker
    local -a limit=()
    batch=$(plan_batch)
    runs="$SEQREC_HOME/runs"
    field() { sed -n "s/^ \"$1\": \"\{0,1\}\([A-Za-z0-9_.]*\)\"\{0,1\},\{0,1\}\$/\1/p" "$2" | sed -n 1p; }
    # Finished runs of this batch, how the last of them ended and the run directory written
    # last. How the plan ended comes first: the description is cut at 140 characters.
    progress() {
        local trained newest last described="" stop epochs wall
        trained=$({ find "$runs/$batch" -name summary.json 2> /dev/null || true; } | wc -l)
        newest=$({ ls -t "$runs/$batch" 2> /dev/null || true; } | sed -n 1p)
        last=$({ ls -t "$runs/$batch"/*/summary.json 2> /dev/null || true; } | sed -n 1p)
        if [ -n "$last" ]; then
            stop=$(field stop_reason "$last") epochs=$(field epochs "$last") wall=$(field wall_s "$last")
            if [[ -n "$stop" && "$epochs" =~ ^[0-9]+$ && "$wall" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                described=", last $(basename "$(dirname "$last")"): $stop, $epochs epochs, $((${wall%.*} / 60)) min"
            fi
        fi
        status "$1" "$TASK $batch${2:+ $2}: $trained runs trained$described, newest ${newest:-none}"
    }
    progress pending
    (while sleep "${PROGRESS_SECONDS:-600}"; do progress pending; done) &
    ticker=$!
    # HOURS bounds the cost of a batch that hangs; a stopped batch resumes when run again.
    # Test reads get no limit: an interrupted read counts as used and cannot be repeated.
    if [ "$TASK" != test ]; then
        limit=(timeout --signal=INT --kill-after=60 "$((HOURS * ${HOUR_SECONDS:-3600}))")
    fi
    # The code under test gets no token.
    "${limit[@]}" env -u GH_TOKEN uv run --no-sync seqrec "$TASK" "configs/plans/$PLAN" \
        --runs-dir "$runs" --data-dir "$SEQREC_HOME/data" || code=$?
    pkill -P "$ticker" 2> /dev/null || true
    kill "$ticker" 2> /dev/null || true
    wait "$ticker" 2> /dev/null || true
    case "$code" in
        0) progress pending "plan finished" ;; # success is reported once the results are pushed
        124) progress failure "stopped at the $HOURS hour limit" ;;
        *) progress failure "exit code $code" ;;
    esac
    return "$code"
}

profile() {
    local -a args
    mapfile -t args < <(profile_args)
    [ "${#args[@]}" -gt 0 ] || exit 1
    for _ in $(seq "$REPEATS"); do
        uv run --no-sync python scripts/profile_step.py --data-dir "$SEQREC_HOME/data" "${args[@]}"
    done
}

# `seqrec data prepare` rewrites results/data; a difference from the committed files means the
# prepared data differ from the data behind the committed results.
data_unchanged() {
    local changed
    changed=$(git status --porcelain -- results/data)
    [ -z "$changed" ] && return 0
    echo "$changed"
    git --no-pager diff -- results/data
    die "results/data differs from the committed files"
}

# Repository code gets no token and no push credentials.
without_token() {
    env -u GH_TOKEN -u GIT_CONFIG_COUNT -u GIT_CONFIG_KEY_0 -u GIT_CONFIG_VALUE_0 "$@"
}

# Put the named paths, and only those, on top of main as it is now and push. Each file
# replaces main's version whole; a CSV that lacks rows main has is refused, so a batch
# exported on another machine cannot be dropped. With `summaries` set to a directory of batch
# files, main's files there must record no test read this machine lacks (their record would
# be replaced), and its overview.csv and tuning_best.csv are computed in the tree that is
# pushed, from every batch file main has at that moment, and committed too.
commit() {
    local message=$1 path lost
    shift
    for path in "$@"; do
        [ -e "$path" ] || die "commit: $path does not exist"
    done
    if [ -n "${GH_TOKEN:-}" ]; then # the checkout keeps no credentials; only this step can push
        export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="http.https://github.com/.extraheader"
        GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"
        export GIT_CONFIG_VALUE_0
    fi
    tree=$(mktemp -d) # not local: the exit trap reads it
    trap 'git worktree remove --force "$tree/main" 2> /dev/null; rm -rf "$tree"' EXIT
    for _ in 1 2 3; do
        if git fetch --quiet origin main; then
            git worktree add --quiet --detach "$tree/main" FETCH_HEAD
            if [ -n "${summaries:-}" ]; then
                without_token uv run --no-sync seqrec export --check-reads \
                    --runs-dir "$SEQREC_HOME/runs" --out-dir "$tree/main/$summaries" ||
                    die "commit: main's $summaries did not pass the check for test reads this machine lacks"
            fi
            for path in "$@"; do
                if [[ "$path" == *.csv && -f "$tree/main/$path" ]]; then
                    lost=$(LC_ALL=C comm -23 <(cut -d, -f1 "$tree/main/$path" | LC_ALL=C sort) \
                        <(cut -d, -f1 "$path" | LC_ALL=C sort) | tr '\n' ' ')
                    [ -z "$lost" ] || die "commit: main's $path has rows this export lacks: $lost"
                fi
                mkdir -p "$(dirname "$tree/main/$path")"
                cp -rT "$path" "$tree/main/$path"
            done
            if [ -n "${summaries:-}" ]; then
                without_token uv run --no-sync seqrec export --summaries-only \
                    --out-dir "$tree/main/$summaries" ||
                    die "commit: the summaries of $summaries could not be computed"
            fi
            git -C "$tree/main" add -- "$@" ${summaries:+"$summaries"}
            if git -C "$tree/main" diff --cached --quiet; then
                echo "nothing to commit"
                return 0
            fi
            git -C "$tree/main" -c user.name="Aaron Yang" -c user.email="yang341133@gmail.com" \
                commit --quiet -m "$message" ${GITHUB_RUN_ID:+-m "Run $GITHUB_RUN_ID at $GITHUB_SHA"}
            if git -C "$tree/main" push ${PUSH_DRY_RUN:+--dry-run} origin HEAD:main; then return 0; fi
            git worktree remove --force "$tree/main"
        fi
        sleep 5
    done
    die "commit: push failed"
}

# pod_start.sh holds this lock for as long as it runs; without it nothing deletes the machine.
watchdog_running() {
    local held=0
    flock -n -E 75 "${POD_START_LOCK:-/run/seqrec-pod-start.lock}" true || held=$?
    case "$held" in
        75) ;; # held
        0) die "the watchdog is not running: nothing would delete this machine" ;;
        *) die "cannot check the watchdog's lock: nothing may be there to delete this machine" ;;
    esac
    pgrep -f "^bash ${POD_START:-/run/seqrec-pod-start.sh}\$" > /dev/null ||
        die "the lock is held, but not by the watchdog: nothing would delete this machine"
}

# Export the volume, once every run of the plan is trained or given up: a run that never
# started has no row, so its batch file would pass for complete. The plan's batch file must
# be among those the export lists as written: the copy that came with the checkout is not
# this machine's result. That copy stays in place for the export, which refuses to replace a
# batch file recording test reads this machine lacks.
export_batch() {
    local batch
    batch=$(plan_batch)
    written=$(mktemp) # not local: the exit trap reads it
    trap 'rm -f "$written"' EXIT
    uv run --no-sync seqrec export --runs-dir "$SEQREC_HOME/runs" --plan "configs/plans/$PLAN" \
        --written "$written"
    grep -qxF "$batch.csv" "$written" || die "export: no run of $batch on this machine"
}

# The plan's batch file and the summaries of results/runs.
results() {
    local batch summaries=results/runs
    batch=$(plan_batch)
    commit "chore(results): add $TASK $batch results" "$summaries/$batch.csv"
}

case "${1:-}" in
    check-inputs) check_inputs ;;
    machine) machine ;;
    status) status "$2" "$3" ;;
    plan) plan ;;
    profile) profile ;;
    data-unchanged) data_unchanged ;;
    watchdog-running) watchdog_running ;;
    export) export_batch ;;
    results) results ;;
    commit) shift && commit "$@" ;;
    *) die "usage: batch_steps.sh watchdog-running|check-inputs|machine|status|plan|profile|data-unchanged|export|results|commit" ;;
esac
