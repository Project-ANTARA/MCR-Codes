#!/usr/bin/env bash
# Run as your normal GitHub-authenticated user, from the repository root.
# Sudo prompts stay in your terminal; registration credentials are never printed.
set +x
set -euo pipefail

if [[ $EUID == 0 ]]; then
    echo 'Run this script as your normal user, not with sudo.' >&2
    exit 1
fi
if [[ $(uname -m) != x86_64 ]]; then
    echo 'This installer is for Linux x86_64 only.' >&2
    exit 1
fi
for cmd in gh curl sha256sum sudo systemctl useradd tar; do
    command -v "$cmd" >/dev/null
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner_account=antara-runner
runner_root=/var/lib/antara-actions
runner_dir=/var/lib/antara-actions/runner
resume=false
if [[ ${1:-} == --resume && $# == 1 ]]; then
    resume=true
elif [[ $# != 0 ]]; then
    echo 'Usage: bash scripts/install_local_runner.sh [--resume]' >&2
    exit 2
fi

sudo -v
if $resume; then
    runner_uid=$(id -u "$runner_account")
    account_home=$(getent passwd "$runner_account" | cut -d: -f6)
    if [[ $runner_uid == 0 || $account_home != "$runner_root" \
          || $(id -Gn "$runner_account") != "$runner_account" ]]; then
        echo 'Existing account does not match the dedicated runner account.' >&2
        exit 1
    fi
    for directory in "$runner_root" "$runner_dir"; do
        if sudo test -L "$directory" || [[ $(sudo stat -c %u "$directory") != "$runner_uid" ]]; then
            echo "Unexpected owner or symlink: $directory" >&2
            exit 1
        fi
    done
    sudo test -x "$runner_dir/config.sh"
    for state in .runner .credentials .service; do
        if sudo test -e "$runner_dir/$state"; then
            echo "Existing $state found; refusing to replace registered/service state." >&2
            exit 1
        fi
    done
    archive_dir='existing verified extraction reused (no download)'
    echo 'Resuming the unregistered runner installation.'
else
    if sudo test -e "$runner_dir" || id "$runner_account" >/dev/null 2>&1; then
        echo 'Installation/account exists. Use --resume after a failed registration.' >&2
        exit 1
    fi

# Version/checksum obtained from GitHub's authenticated runner-downloads API.
archive_dir=$(mktemp -d /tmp/antara-runner-download-XXXXXX)
curl --fail --location --retry 3 \
    https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-linux-x64-2.337.0.tar.gz \
    --output "$archive_dir/runner.tar.gz"
printf '%s  %s\n' \
    70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613 \
    "$archive_dir/runner.tar.gz" | sha256sum --check -

sudo useradd --system --user-group --create-home --home-dir "$runner_root" \
    --shell /bin/bash "$runner_account"
sudo install -d -m 0750 -o "$runner_account" -g "$runner_account" "$runner_dir"
sudo -H -u "$runner_account" tar -xz -C "$runner_dir" < "$archive_dir/runner.tar.gz"
fi

runner_token=$(gh api --method POST repos/Project-ANTARA/MCR-Codes/actions/runners/registration-token --jq .token)
if [[ -z $runner_token || $runner_token == null ]]; then
    echo 'GitHub did not return a registration token.' >&2
    exit 1
fi
sudo -v
printf '%s\n' "$runner_token" | sudo -n -H -u "$runner_account" \
    env PATH=/usr/local/bin:/usr/bin:/bin bash -c '
        set -euo pipefail
        IFS= read -r ACTIONS_RUNNER_INPUT_TOKEN
        export ACTIONS_RUNNER_INPUT_TOKEN
        cd /var/lib/antara-actions/runner
        ./config.sh --unattended --url https://github.com/Project-ANTARA/MCR-Codes \
            --name antara-local-nightly --labels antara-nightly --work _work
    '
unset runner_token

# svc.sh requires its own working directory; invoke it as administrator there.
sudo bash -c 'cd /var/lib/antara-actions/runner && ./svc.sh install antara-runner'
service_name=$(sudo cat "$runner_dir/.service")
if [[ ! "$service_name" =~ ^actions\.runner\.[A-Za-z0-9_.-]+\.service$ ]]; then
    echo 'Unexpected runner service name; refusing to install override.' >&2
    exit 1
fi
sudo install -d -m 0755 "/etc/systemd/system/$service_name.d"
sudo install -m 0644 "$script_dir/antara-runner-hardening.conf" \
    "/etc/systemd/system/$service_name.d/antara-hardening.conf"
sudo systemctl daemon-reload
sudo systemctl enable --now "$service_name"
sudo systemctl --no-pager status "$service_name"
gh api repos/Project-ANTARA/MCR-Codes/actions/runners \
    --jq '.runners[] | select(.name == "antara-local-nightly") | {name,status,labels:[.labels[].name]}'
printf '\nRunner service: %s\nDownload retained at: %s\n' "$service_name" "$archive_dir"
echo 'No workflow was dispatched. Keep this machine awake and online for nightly jobs.'
