#!/usr/bin/env bash
# Steps of the batch workflow (.github/workflows/batch.yml). Inputs arrive as environment
# variables and are checked here before anything uses them.
#   check-inputs | status STATE TEXT | plan | profile | data-unchanged | commit MESSAGE PATH...
set -euo pipefail

DATASETS="ml-1m ml-20m steam video-games"

die() {
    echo "$*" >&2
    exit 1
}

# The batch a plan file belongs to; the plan must be named after it.
plan_batch() {
    [[ "${PLAN:-}" =~ ^[A-Za-z0-9_-]+\.toml$ ]] || die "plan: not a plan file name"
    [ -f "configs/plans/$PLAN" ] || die "plan: configs/plans/$PLAN does not exist"
    grep -qxF "batch = \"${PLAN%.toml}\"" "configs/plans/$PLAN" ||
        die "plan: the batch in $PLAN is not named ${PLAN%.toml}"
    echo "${PLAN%.toml}"
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
        run | tune | test)
            plan_batch > /dev/null
            [[ "${HOURS:-}" =~ ^([1-9]|1[0-9]|2[0-3])$ ]] || die "hours: a whole number from 1 to 23"
            ;;
        profile)
            profile_args > /dev/null
            [[ "${REPEATS:-}" =~ ^([1-9]|1[0-9]|20)$ ]] || die "repeats: a whole number from 1 to 20"
            ;;
        *) die "task: unknown" ;;
    esac
}

# Commit status "batch/progress" on the commit the job runs from. The job log is readable
# only after the job ends; this is readable while it runs. Never fails the step.
status() {
    local text
    text=$(printf '%s' "$2" | tr -cd 'A-Za-z0-9 ._:,/-' | cut -c1-140)
    curl -fsS -m 20 -o /dev/null -H "Authorization: Bearer $GH_TOKEN" \
        "$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/statuses/$GITHUB_SHA" \
        -d "{\"state\":\"$1\",\"context\":\"batch/progress\",\"description\":\"$text\"}" || true
}

plan() {
    local batch runs code=0 ticker
    batch=$(plan_batch)
    runs="$SEQREC_HOME/runs"
    progress() { # finished runs of this batch and the run directory written last
        local trained newest
        trained=$({ find "$runs/$batch" -name summary.json 2> /dev/null || true; } | wc -l)
        newest=$({ ls -t "$runs/$batch" 2> /dev/null || true; } | sed -n 1p)
        status "$1" "$TASK $batch: $trained runs trained, newest ${newest:-none}${2:+, $2}"
    }
    progress pending
    (while sleep "${PROGRESS_SECONDS:-300}"; do progress pending; done) &
    ticker=$!
    # HOURS bounds the cost of a batch that hangs; a stopped batch resumes when run again.
    timeout --signal=INT --kill-after=60 "$((HOURS * ${HOUR_SECONDS:-3600}))" \
        uv run --no-sync seqrec "$TASK" "configs/plans/$PLAN" \
        --runs-dir "$runs" --data-dir "$SEQREC_HOME/data" || code=$?
    kill "$ticker" 2> /dev/null || true
    case "$code" in
        0) progress success ;;
        124 | 137) progress failure "stopped at the $HOURS hour limit" ;;
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

# Commit the named paths, and only those, on top of main as it is now, then push.
commit() {
    local message=$1 path
    shift
    for path in "$@"; do
        if [ -e "$path" ]; then git add -- "$path"; fi
    done
    if git diff --cached --quiet; then
        echo "nothing to commit"
        return 0
    fi
    git -c user.name="$(git log -1 --format=%an)" -c user.email="$(git log -1 --format=%ae)" \
        commit --quiet -m "$message"
    if [ -n "${GH_TOKEN:-}" ]; then # the checkout keeps no credentials; only this step can push
        export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="http.https://github.com/.extraheader"
        GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"
        export GIT_CONFIG_VALUE_0
    fi
    git stash --quiet --include-untracked # what the export wrote for other batches
    for _ in 1 2 3; do
        git fetch --quiet origin main
        if ! git -c user.name=batch -c user.email=batch@localhost rebase --quiet FETCH_HEAD; then
            git rebase --abort
            die "conflict with main: the results stay in the runs directory, nothing was pushed"
        fi
        if git push ${PUSH_DRY_RUN:+--dry-run} origin HEAD:main; then return 0; fi
        sleep 5
    done
    die "push failed"
}

case "${1:-}" in
    check-inputs) check_inputs ;;
    status) status "$2" "$3" ;;
    plan) plan ;;
    profile) profile ;;
    data-unchanged) data_unchanged ;;
    commit) shift && commit "$@" ;;
    *) die "usage: batch_steps.sh check-inputs|status|plan|profile|data-unchanged|commit" ;;
esac
