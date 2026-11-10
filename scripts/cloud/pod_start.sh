#!/usr/bin/env bash
# The machine's watchdog. The container start command runs it at every start and again
# whenever it exits. It keeps the GitHub Actions runner online and stops the machine when a
# job asks for it, when no job has run for IDLE_MINUTES, or when one job has run longer than
# any job may. Only /workspace survives a stop.
set -uo pipefail

HOME_DIR=${SEQREC_HOME:-/workspace/seqrec}
RUNNER_DIR=${RUNNER_DIR:-/workspace/actions-runner}
STOP_REQUEST=${STOP_REQUEST:-/run/seqrec-stop-request} # written by the workflow's last step
LOCK=${POD_START_LOCK:-/run/seqrec-pod-start.lock}
CHECK_SECONDS=${CHECK_SECONDS:-60}
IDLE_MINUTES=${IDLE_MINUTES:-30}
BUSY_MINUTES=${BUSY_MINUTES:-1500} # the workflow ends a job after 23 hours

log() { echo "$(date -u +%FT%TZ) $*"; }

exec 9> "$LOCK" || { log "cannot open $LOCK"; exit 1; }
flock -n 9 || exit 0 # another instance is running

# A shell opened later may lack the variables the container was started with.
from_init() { tr '\0' '\n' < /proc/1/environ 2> /dev/null | sed -n "s/^$1=//p" | head -1; }
export RUNPOD_POD_ID=${RUNPOD_POD_ID:-$(from_init RUNPOD_POD_ID)}
export RUNPOD_API_KEY=${RUNPOD_API_KEY:-$(from_init RUNPOD_API_KEY)}

# Both ways, every time: a stop that worked ends this script before the next line.
stop_machine() {
    log "stopping the machine: $1"
    timeout 60 runpodctl pod stop "$RUNPOD_POD_ID" 9>&-
    sleep "${STOP_WAIT_SECONDS:-20}" 9>&-
    curl -fsS -m 30 -X POST "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" \
        -H @- -H "Content-Type: application/json" -d '{"action":"stop"}' 9>&- \
        <<< "Authorization: Bearer $RUNPOD_API_KEY"
    log "STILL RUNNING AFTER TWO STOP ATTEMPTS; trying again at the next check"
}

listener() { pgrep -f "$RUNNER_DIR/(run\.sh|run-helper|bin[^/]*/Runner\.Listener)" > /dev/null; }
worker() { pgrep -of "$RUNNER_DIR/bin[^/]*/Runner\.Worker"; } # prints the pid of a running job

start_runner() {
    [ -f "$RUNNER_DIR/.runner" ] || return 0 # not registered yet
    listener && return 0
    log "starting the runner"
    # System libraries are lost with the container disk.
    DEBIAN_FRONTEND=noninteractive timeout 600 "$RUNNER_DIR/bin/installdependencies.sh" \
        < /dev/null > /dev/null 2>&1 9>&-
    # Jobs get uv on PATH. The machine's API key stays out of their environment so that it
    # cannot end up in a log or a report; a job runs as root and could still read it.
    (cd "$RUNNER_DIR" && exec env -u RUNPOD_API_KEY -u PUBLIC_KEY -u JUPYTER_PASSWORD \
        PATH="$HOME_DIR/bin:$PATH" RUNNER_ALLOW_RUNASROOT=1 ./run.sh) < /dev/null 9>&- &
}

[ -n "$RUNPOD_POD_ID" ] || log "RUNPOD_POD_ID IS NOT SET: THIS SCRIPT CANNOT STOP THE MACHINE"
idle=0 busy=0 asked=0 job=""
while true; do
    start_runner
    sleep "$CHECK_SECONDS" 9>&- || true # no child keeps the lock: it must die with this script
    if now=$(worker); then
        [ "$now" = "$job" ] || busy=0 # another job than at the last check
        job=$now idle=0 asked=0 busy=$((busy + 1))
        if [ "$busy" -ge "$BUSY_MINUTES" ]; then stop_machine "one job has run for $busy checks"; fi
        continue
    fi
    job="" busy=0 idle=$((idle + 1))
    if [ -e "$STOP_REQUEST" ]; then asked=$((asked + 1)); else asked=0; fi
    # Two checks in a row: the job has reported by then, and a job queued behind it has started.
    if [ "$asked" -ge 2 ]; then
        stop_machine "a job asked for it"
    elif [ "$idle" -ge "$IDLE_MINUTES" ]; then
        stop_machine "no job for $idle checks"
    fi
done
