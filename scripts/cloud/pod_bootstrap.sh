#!/usr/bin/env bash
# One-time setup of a rented container as a GitHub Actions runner for this repository.
# Run it in the machine's terminal; it asks for the runner registration token. It can be run
# again, for example after GitHub has removed a runner that was offline for 14 days.
#   bash pod_bootstrap.sh
# The machine must have been created from configs/cloud/pod.json: its start command runs the
# watchdog, which deletes the machine after IDLE_MINUTES (30 unless the machine was created with
# another value) without a job, whether or not this script succeeds.
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
    mountpoint -q /workspace || die "/workspace is not a volume: nothing would outlive this machine. Delete this machine by hand."
    mkdir -p "$HOME_DIR/bin" "$RUNNER_DIR"
    # What the watchdog needs; without any of these it could not delete the machine.
    for tool in curl pgrep flock timeout; do
        command -v "$tool" > /dev/null || die "$tool is missing. Delete this machine by hand."
    done
    local name
    for name in RUNPOD_POD_ID RUNPOD_API_KEY; do
        [ -n "${!name:-$(tr '\0' '\n' < /proc/1/environ 2> /dev/null | sed -n "s/^$name=//p")}" ] ||
            die "$name is not set: the machine could not delete itself. Delete this machine by hand."
    done
    local held=0
    flock -n -E 75 "$LOCK" true || held=$?
    { [ "$held" -eq 75 ] && pgrep -f "^bash ${POD_START:-/run/seqrec-pod-start.sh}\$" > /dev/null; } ||
        die "the watchdog is not running: nothing would delete this machine. Delete it by hand."
    echo "Watchdog running: this machine deletes itself when it has been without a job for its idle limit."

    command -v tar > /dev/null || die "tar is missing"
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
