#!/usr/bin/env bash
# One-time setup of a rented GPU container as a GitHub Actions runner for this repository.
# Run it in the machine's terminal; it asks for the runner registration token. It can be run
# again, for example after GitHub has removed a runner that was offline for 14 days.
#   bash pod_bootstrap.sh
# Once it has printed "Watchdog started" the machine stops itself after 30 minutes without a
# job, whether or not the rest of this script succeeds.
set -euo pipefail

REPO_URL=https://github.com/sHyanorn/seqrec-lab
HOME_DIR=${SEQREC_HOME:-/workspace/seqrec}
RUNNER_DIR=${RUNNER_DIR:-/workspace/actions-runner}
UV_VERSION=0.11.32
LABEL=gpu # the batch workflow runs on this label
LOCK=${POD_START_LOCK:-/run/seqrec-pod-start.lock} # held by pod_start.sh while it runs

die() {
    echo "$*" >&2
    exit 1
}

main() {
    mountpoint -q /workspace || die "/workspace is not a volume: nothing would survive a stop. Stop this machine by hand."
    mkdir -p "$HOME_DIR/bin" "$RUNNER_DIR"
    curl -fsSL "$REPO_URL/raw/main/scripts/cloud/pod_start.sh" -o "$HOME_DIR/pod_start.sh.new"
    chmod +x "$HOME_DIR/pod_start.sh.new"
    mv "$HOME_DIR/pod_start.sh.new" "$HOME_DIR/pod_start.sh"
    # What the watchdog needs; without any of these it could not stop the machine.
    for tool in runpodctl pgrep pkill flock timeout; do
        command -v "$tool" > /dev/null || die "$tool is missing. Stop this machine by hand."
    done
    [ -n "${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2> /dev/null | sed -n 's/^RUNPOD_POD_ID=//p')}" ] ||
        die "RUNPOD_POD_ID is not set: the machine could not stop itself. Stop this machine by hand."
    # An older copy of the watchdog, and only that: the start command's loop names the same path.
    local watchdog="^bash $HOME_DIR/pod_start\.sh\$" waited=0 held
    pkill -f "$watchdog" || true
    nohup "$HOME_DIR/pod_start.sh" >> "$HOME_DIR/runner.log" 2>&1 < /dev/null &
    while true; do
        sleep 1 # first: a probe that beat the watchdog to the lock would make it exit
        held=0
        flock -n -E 75 "$LOCK" true || held=$?
        case "$held" in
            75) break ;; # held
            0) ;; # free: no watchdog holds it yet
            *) die "the watchdog's lock cannot be checked. Stop this machine by hand." ;;
        esac
        waited=$((waited + 1))
        [ "$waited" -lt 20 ] || die "the watchdog did not start. Stop this machine by hand."
    done
    pgrep -f "$watchdog" > /dev/null ||
        die "the lock is held, but not by the watchdog. Stop this machine by hand."
    echo "Watchdog started: this machine stops itself after 30 minutes without a job."

    for tool in tar nvidia-smi; do
        command -v "$tool" > /dev/null || die "$tool is missing"
    done
    # The loop of the container start command; without it nothing runs after the next start.
    pgrep -f "while true; do \[ -x $HOME_DIR/pod_start.sh" > /dev/null ||
        die "the container start command is not set as documented: stop this machine and fix it"
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), driver $driver"
    [ "${driver%%.*}" -ge 580 ] || die "driver branch below 580: the CUDA 13 wheels will not run here"

    local token=${RUNNER_TOKEN:-}
    if [ -z "$token" ]; then
        read -rsp "Runner registration token: " token < /dev/tty
        echo
    fi
    [[ "$token" =~ ^[A-Z0-9]{20,}$ ]] || die "that does not look like a registration token"

    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" |
        env UV_INSTALL_DIR="$HOME_DIR/bin" UV_NO_MODIFY_PATH=1 sh
    "$HOME_DIR/bin/uv" --version

    cd "$RUNNER_DIR"
    local tag
    tag=$(basename "$(curl -fsSLI -o /dev/null -w '%{url_effective}' https://github.com/actions/runner/releases/latest)")
    [[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "could not read the latest runner version"
    curl -fsSL "https://github.com/actions/runner/releases/download/$tag/actions-runner-linux-x64-${tag#v}.tar.gz" -o runner.tar.gz
    echo "runner $tag, sha256 $(sha256sum runner.tar.gz | cut -d' ' -f1)"
    tar xzf runner.tar.gz
    rm runner.tar.gz
    DEBIAN_FRONTEND=noninteractive ./bin/installdependencies.sh < /dev/null
    rm -f .runner .credentials .credentials_rsaparams # a registration from an earlier run of this script
    # The token goes in through the environment, not the command line.
    RUNNER_ALLOW_RUNASROOT=1 ACTIONS_RUNNER_INPUT_TOKEN="$token" ./config.sh --unattended \
        --url "$REPO_URL" --name "$LABEL" --labels "$LABEL" --replace < /dev/null
    echo "Registered. Within two minutes the runner should show as Idle under Settings > Actions > Runners."
}

main "$@" # the whole script is read before any of it runs
