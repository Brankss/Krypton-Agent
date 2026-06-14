# Deploy Krypton on an always-on cloud VM (Oracle Cloud Free Tier)

This runs Krypton's **Telegram bot** 24/7 on a free Oracle Cloud VM, with the
code **auto-synced from GitHub**: every couple of minutes the VM pulls the
latest commit of its branch and, if anything changed, redeploys and restarts
itself. You keep developing on GitHub; the cloud follows automatically.

> **No inbound ports needed.** The bot talks to Telegram over *outbound*
> long-polling, so you don't open any firewall ports or expose a public IP —
> only SSH (already open on Oracle) is used, and only by you.

---

## What you get

| Unit | Role |
|---|---|
| `krypton.service` | Runs `python -m krypton telegram`. Auto-restarts on crash, starts on boot. |
| `krypton-sync.service` + `.timer` | Every 2 min: `git fetch` → if the branch moved, `reset --hard`, reinstall deps *only if `pyproject.toml` changed*, restart the bot. |
| `.github/workflows/ci.yml` | Runs ruff + the test suite on every push/PR, so what the VM pulls is already green. |

A restart mid-conversation is safe — Krypton persists each chat's context to
SQLite and restores it on the next message.

---

## 1. Create the instance (Oracle Console)

1. **Compute → Instances → Create instance.**
2. **Shape:** `VM.Standard.A1.Flex` (Ampere ARM — the "huge" Always-Free one;
   e.g. 2 OCPU / 12 GB is plenty). x86 micro shapes work too but are tiny.
3. **Image:** Ubuntu 24.04 or Oracle Linux 9 (both ship Python ≥ 3.11).
4. **SSH:** upload/keep your public key so you can log in.
5. Create it and note the **public IP**.

You do **not** need to touch the VCN security lists — outbound is open by
default and the bot needs nothing inbound except SSH (already allowed).

## 2. Install (one script)

SSH in, then:

```bash
ssh ubuntu@<PUBLIC_IP>        # or opc@<IP> on Oracle Linux

git clone https://github.com/Brankss/Krypton-Agent.git
cd Krypton-Agent
bash deploy/install.sh
```

The installer is **idempotent** — re-run it any time to repair or upgrade.
It installs Python, creates a venv, installs Krypton, writes the systemd
units, and enables the auto-sync timer.

> **Tracking `main`:** by default the VM tracks the `main` branch. Make sure
> your latest code is merged to `main` first. To track a different branch for
> testing, run: `KRYPTON_DEPLOY_BRANCH=my-branch bash deploy/install.sh`.

## 3. Add your secrets

The installer creates `~/Krypton-Agent/.env` from `deploy/env.example` but
leaves the secrets blank, so the bot won't start until you fill them:

```bash
nano ~/Krypton-Agent/.env
```

Set at minimum:

```ini
KRYPTON_PROVIDER=nvidia
NVIDIA_API_KEY=<free key from build.nvidia.com>

TELEGRAM_BOT_TOKEN=<from @BotFather>
TELEGRAM_AUTHORIZED_USER_IDS=<your numeric id from @userinfobot>
```

Then start it:

```bash
sudo systemctl start krypton
```

## 4. Verify

```bash
sudo systemctl status krypton      # should say active (running)
journalctl -u krypton -f           # live logs
```

Open Telegram → your bot → `/start`. You're live, 24/7.

---

## How the auto-sync works

`krypton-sync.timer` runs `deploy/sync.sh` every 2 minutes (configurable). It:

1. `git fetch` the branch this checkout is on,
2. if the remote tip moved → `git reset --hard` to it
   (your `.env`, `data/`, and `inbox/` are gitignored and **preserved**),
3. reinstalls deps **only if `pyproject.toml` changed**,
4. restarts `krypton.service`.

So your workflow is simply: **push to `main` → within ~2 min the VM is running
the new code.**

Change the cadence by editing `OnUnitActiveSec` in
`/etc/systemd/system/krypton-sync.timer` (then `sudo systemctl daemon-reload`).

### Want *instant* deploys instead of a 2-min poll? (optional)

Add a GitHub Actions job that SSHes in on push and runs the sync. You'd add
repo **secrets** `SSH_HOST`, `SSH_USER`, `SSH_KEY` (a private key whose public
half is in the VM's `~/.ssh/authorized_keys`) and a workflow step like:

```yaml
- uses: appleboy/ssh-action@v1
  with:
    host: ${{ secrets.SSH_HOST }}
    username: ${{ secrets.SSH_USER }}
    key: ${{ secrets.SSH_KEY }}
    script: bash ~/Krypton-Agent/deploy/sync.sh
```

The 2-min timer already keeps you in sync, so this is purely a latency upgrade.

---

## Private repo? Use a read-only deploy key

If `Brankss/Krypton-Agent` is private, anonymous HTTPS clone/fetch fails. Give
the VM a **read-only deploy key**:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/krypton_deploy -N ""
cat ~/.ssh/krypton_deploy.pub   # add this on GitHub:
# Repo → Settings → Deploy keys → Add deploy key (read-only is enough)

cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/krypton_deploy
  IdentitiesOnly yes
EOF

# then install against the SSH URL:
KRYPTON_REPO_URL=git@github.com:Brankss/Krypton-Agent.git bash deploy/install.sh
```

---

## Choosing a model on the VM

- **NVIDIA NIM (recommended):** free, no GPU needed, no per-day cap. Set
  `KRYPTON_PROVIDER=nvidia` + `NVIDIA_API_KEY`. Zero load on the VM.
- **OpenRouter / Ollama Cloud:** other hosted options (set the matching keys).
- **Local Ollama on the VM:** the Ampere A1 can run small models on CPU.
  `curl -fsSL https://ollama.com/install.sh | sh`, `ollama pull qwen2.5-coder:3b`,
  set `KRYPTON_PROVIDER=ollama_local`. Slower, but fully private and free.

---

## Operations cheat-sheet

```bash
sudo systemctl status krypton              # is the bot up?
sudo systemctl restart krypton             # manual restart
journalctl -u krypton -f                   # live bot logs
journalctl -u krypton-sync -n 30           # recent auto-deploys
systemctl list-timers krypton-sync.timer   # when's the next sync
bash ~/Krypton-Agent/deploy/sync.sh        # force a sync now
```

### Uninstall

```bash
sudo systemctl disable --now krypton.service krypton-sync.timer
sudo rm /etc/systemd/system/krypton.service \
        /etc/systemd/system/krypton-sync.service \
        /etc/systemd/system/krypton-sync.timer \
        /etc/sudoers.d/krypton-restart
sudo systemctl daemon-reload
# rm -rf ~/Krypton-Agent   # also removes data/ (memory db) and .env
```
