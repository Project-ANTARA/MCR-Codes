# Local full-year nightly runner

## Status and installation

The workflow now targets a dedicated Linux x64 runner labeled `antara-nightly`.
Installation and connectivity were **verified on 2026-09-13** after the user
completed the sudo installation. GitHub reported `antara-local-nightly` online
and idle, with labels `self-hosted`, `Linux`, `X64`, and `antara-nightly`.
The service logged `Connected to GitHub` at 14:03:35 IST and `Listening for Jobs`
at 14:03:36 IST (08:33:36 UTC). The installer's initial `offline` response was
a startup timing effect, not a registration failure. The repository was private
at inspection. No workflow was dispatched during setup verification.

The systemd service was enabled and active/running with zero restarts, running
as `antara-runner`. Verified protections: `ProtectHome=yes`,
`ProtectSystem=strict`, and `NoNewPrivileges=yes`. These checks establish runner
connectivity, **not** successful environment creation or full-year simulation.

## Fresh installation and recovery

The existing machine is already installed; do not rerun installation or resume
on the healthy registered runner. For a new installation, run from the repository
root as your normal GitHub-authenticated user:

```bash
bash scripts/install_local_runner.sh
```

If the first installer stopped with `Invalid configuration provided for token`,
the corrected script can reuse the downloaded/extracted runner:

```bash
bash scripts/install_local_runner.sh --resume
```

The original installer used the wrong environment-variable prefix. The runner
requires `ACTIONS_RUNNER_INPUT_TOKEN`. Resume obtains a fresh token without
printing it, checks the dedicated account and directory ownership, and refuses
to overwrite existing registration, credentials, or service state. Do not
delete the account or installation to recover from this error.

Enter your sudo password only in your terminal, never in chat. The installer:

- Downloads official runner 2.337.0 and verifies its API-provided SHA-256.
- Creates the unprivileged `antara-runner` account under `/var/lib/antara-actions`.
- Registers `antara-local-nightly` with label `antara-nightly`; the short-lived
  registration token is piped privately and not saved in the repository.
- Installs/enables the official systemd service with a hardening override that
  hides personal home directories and limits persistent writes to runner state.
- Leaves runner auto-updates enabled and does not dispatch any workflow.

The account has no sudo privileges. Runner files live under its own home;
workflow checkouts and the planned Micromamba Python 3.11 `antara` environment
use that separate account, away from the developer's Conda `orekit_env`.
Ordinary local development still uses Conda.
The service uses GitHub's `runsvc.sh` entrypoint. See [GitHub service guidance](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application).

Arch Linux was detected locally. Runner 2.337.0 started successfully, confirming
its startup dependencies. Workflow Micromamba installation and physics backend
execution still require verification under the service account. The installer
does not automatically upgrade system packages. On failure, retain the error
and ask for diagnosis; existing accounts/directories are deliberately not
overwritten on rerun. Downloads remain in the printed temporary directory.

## Execution policy and limits

Only scheduled nightly or manual `suite=nightly` runs on the repository's
default branch target this runner. Push, PR, and weekly validation jobs stay
GitHub-hosted. Nightly jobs share a non-cancelling concurrency group across
scheduled/manual runs to serialize use of this machine.

The full 8,760-hour preset, multirate stepping, physics assertions, plotting,
and full-duration telemetry checks remain unchanged. The annual process gets
20 wall hours via GNU `timeout` (timeouts fail the job); the whole job gets
23 hours, leaving nominal room for setup, both 24-hour scenarios, 5,000 Monte
Carlo trials, validation and uploads. This is an initial operational budget,
**not a verified full-year runtime**. GitHub caps an individual step's
`timeout-minutes` at 360, so the annual step inherits the longer job deadline
and uses the shell timeout. Staying below 24 hours avoids introducing another
credential for long-lived API access. See [workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
and [token lifetime](https://docs.github.com/en/actions/concepts/security/github_token).

Keep the machine powered, awake, on AC and connected. No suspend/lid settings
were modified. At inspection there were approximately 11 GiB available RAM
and 215 GiB free disk; full-year peak memory and runtime have not been measured.
Avoid competing heavy jobs. Missing/offline runners queue jobs rather than
falling back to a shortened simulation. Long runs may cause pending nightly
requests to be superseded; this is not a guarantee every daily request executes.

Branch guards protect this workflow, not every workflow a collaborator could
add. Keep this runner limited to trusted repository maintainers and private
code. Reassess isolation before making the repository public; do not use it
for untrusted pull requests.

## Verification and service control

```bash
gh api repos/Project-ANTARA/MCR-Codes/actions/runners \
  --jq '.runners[] | {name,status,labels:[.labels[].name]}'
systemctl status actions.runner.Project-ANTARA-MCR-Codes.antara-local-nightly.service
sudo journalctl -u actions.runner.Project-ANTARA-MCR-Codes.antara-local-nightly.service -n 80 --no-pager
```

After the workflow changes are published on the default branch, a maintainer
can explicitly launch `gh workflow run ci.yml --ref main -f suite=nightly`.
That launches the real full-year campaign; the assistant did not publish or
dispatch it during setup verification. Check the default-branch workflow before
dispatching; an online runner alone does not establish that routing is deployed.
Use the exact service name printed by the installer with `sudo systemctl stop`
or `sudo systemctl disable --now` to stop it. Preserve run outputs before
reinstalling or removing the service/account.
