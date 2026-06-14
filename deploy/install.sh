#!/usr/bin/env bash
#
# Krypton Agent — one-shot installer for an always-on cloud VM.
# Target: Oracle Cloud Always-Free (Ampere A1 / x86), Ubuntu or Oracle Linux.
#
# Idempotent: safe to re-run to repair or upgrade an existing install.
#
# It sets up:
#   krypton.service       — runs the Telegram bot, auto-restart, starts on boot
#   krypton-sync.service  — git pull + redeploy when the tracked branch moves
#   krypton-sync.timer    — fires the sync on a schedule (default: every 2 min)
#
# The Telegram bot uses outbound long-polling, so NO inbound firewall ports
# are required — only SSH (already open on Oracle).
#
# Usage (from a fresh clone — recommended):
#   git clone https://github.com/Brankss/Krypton-Agent.git
#   cd Krypton-Agent && bash deploy/install.sh
#
# Overridable via environment:
#   KRYPTON_REPO_URL    (default: the public HTTPS URL below)
#   KRYPTON_DEPLOY_BRANCH (default: main)
#   KRYPTON_APP_DIR     (default: this checkout, else $HOME/krypton-agent)
#   KRYPTON_SYNC_INTERVAL (default: 2min)
#
set -euo pipefail

REPO_URL="${KRYPTON_REPO_URL:-https://github.com/Brankss/Krypton-Agent.git}"
DEPLOY_BRANCH="${KRYPTON_DEPLOY_BRANCH:-main}"
SYNC_INTERVAL="${KRYPTON_SYNC_INTERVAL:-2min}"
RUN_USER="$(id -un)"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$RUN_USER" = "root" ] && die "Run this as your normal login user (opc/ubuntu), not root — it will sudo only where needed."
command -v sudo >/dev/null || die "sudo is required."

# --- decide where the app lives -------------------------------------------
# If this script is run from inside an existing checkout, use it in place;
# otherwise clone fresh into $HOME/krypton-agent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
MAYBE_REPO="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd || echo "")"
if [ -n "$MAYBE_REPO" ] && [ -d "$MAYBE_REPO/.git" ] && [ -f "$MAYBE_REPO/pyproject.toml" ]; then
  APP_DIR="${KRYPTON_APP_DIR:-$MAYBE_REPO}"
else
  APP_DIR="${KRYPTON_APP_DIR:-$HOME/krypton-agent}"
fi

# --- 1. system dependencies -----------------------------------------------
say "Installing system dependencies"
if command -v apt-get >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y git curl ca-certificates build-essential || true
  sudo apt-get install -y python3.12 python3.12-venv python3.12-dev 2>/dev/null \
    || sudo apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null \
    || sudo apt-get install -y python3 python3-venv python3-dev
elif command -v dnf >/dev/null; then
  sudo dnf install -y git curl gcc make 2>/dev/null || true
  sudo dnf install -y python3.12 python3.12-pip 2>/dev/null \
    || sudo dnf install -y python3.11 python3.11-pip 2>/dev/null \
    || sudo dnf install -y python3 python3-pip
else
  die "Unsupported distro (need apt or dnf). Install git + python>=3.11 manually, then re-run."
fi

# --- pick a python >= 3.11 ------------------------------------------------
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null; then
    v="$("$cand" -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])' 2>/dev/null || echo 0)"
    if [ "$v" -ge 311 ]; then PY="$(command -v "$cand")"; break; fi
  fi
done
[ -n "$PY" ] || die "No python >= 3.11 found. Install it (Ubuntu 24.04 / Oracle Linux 9 have it) and re-run."
say "Python: $PY ($("$PY" --version 2>&1))"

# --- 2. clone or update repo ----------------------------------------------
if [ -d "$APP_DIR/.git" ]; then
  say "Updating checkout in $APP_DIR (branch $DEPLOY_BRANCH)"
  git -C "$APP_DIR" fetch --prune origin
  git -C "$APP_DIR" checkout "$DEPLOY_BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$DEPLOY_BRANCH"
else
  say "Cloning $REPO_URL -> $APP_DIR (branch $DEPLOY_BRANCH)"
  git clone --branch "$DEPLOY_BRANCH" "$REPO_URL" "$APP_DIR" \
    || die "Clone failed. If the repo is private, set up a read-only deploy key (see deploy/README.md) and re-run with KRYPTON_REPO_URL=git@github.com:Brankss/Krypton-Agent.git"
fi

# --- 3. venv + install -----------------------------------------------------
say "Creating virtualenv + installing Krypton"
"$PY" -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q -U pip wheel
"$APP_DIR/.venv/bin/pip" install -q -e "$APP_DIR"
mkdir -p "$APP_DIR/data"

# --- 4. .env ---------------------------------------------------------------
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  say "Creating $ENV_FILE from deploy/env.example"
  sed "s#__APP_DIR__#$APP_DIR#g" "$APP_DIR/deploy/env.example" > "$ENV_FILE"
  chmod 600 "$ENV_FILE"
else
  say ".env already present — leaving your secrets untouched"
fi

# --- 5. systemd units ------------------------------------------------------
say "Installing systemd units"
SYSTEMCTL="$(command -v systemctl)"
VENV_PY="$APP_DIR/.venv/bin/python"

sudo tee /etc/systemd/system/krypton.service >/dev/null <<UNIT
[Unit]
Description=Krypton Agent (Telegram bot)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
ExecStart=$VENV_PY -m krypton telegram
Restart=always
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
UNIT

sudo tee /etc/systemd/system/krypton-sync.service >/dev/null <<UNIT
[Unit]
Description=Krypton Agent — pull latest from GitHub and redeploy if changed
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$RUN_USER
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/env bash $APP_DIR/deploy/sync.sh
UNIT

sudo tee /etc/systemd/system/krypton-sync.timer >/dev/null <<UNIT
[Unit]
Description=Run krypton-sync every $SYNC_INTERVAL

[Timer]
OnBootSec=1min
OnUnitActiveSec=$SYNC_INTERVAL
Unit=krypton-sync.service

[Install]
WantedBy=timers.target
UNIT

# --- 6. sudoers: allow the sync to restart ONLY krypton.service -----------
say "Allowing $RUN_USER to restart krypton.service (scoped sudoers rule)"
echo "$RUN_USER ALL=(root) NOPASSWD: $SYSTEMCTL restart krypton.service" \
  | sudo tee /etc/sudoers.d/krypton-restart >/dev/null
sudo chmod 440 /etc/sudoers.d/krypton-restart

# --- 7. enable -------------------------------------------------------------
say "Enabling services"
sudo "$SYSTEMCTL" daemon-reload
sudo "$SYSTEMCTL" enable krypton.service >/dev/null 2>&1 || true
sudo "$SYSTEMCTL" enable --now krypton-sync.timer >/dev/null 2>&1 || true

STARTED=0
if grep -Eq '^[[:space:]]*TELEGRAM_BOT_TOKEN=.+' "$ENV_FILE"; then
  sudo "$SYSTEMCTL" restart krypton.service && STARTED=1
fi

# --- summary ---------------------------------------------------------------
printf '\n──────────────────────────────────────────────────────────────────────\n'
echo " Krypton installed at : $APP_DIR"
echo " Auto-sync            : every $SYNC_INTERVAL from branch '$DEPLOY_BRANCH'"
printf '──────────────────────────────────────────────────────────────────────\n'
if [ "$STARTED" = "1" ]; then
  echo " Bot status: RUNNING.  ->  sudo systemctl status krypton"
else
  cat <<EOF
 Bot status: NOT STARTED — fill in your secrets first:

   nano $APP_DIR/.env
       TELEGRAM_BOT_TOKEN=...            (from @BotFather)
       TELEGRAM_AUTHORIZED_USER_IDS=...  (your id from @userinfobot)
       NVIDIA_API_KEY=...                (free at build.nvidia.com)

   sudo systemctl start krypton
EOF
fi
cat <<EOF

 Handy:
   sudo systemctl status krypton             # bot health
   journalctl -u krypton -f                  # live bot logs
   journalctl -u krypton-sync -n 30          # recent auto-deploys
   systemctl list-timers krypton-sync.timer  # next sync
──────────────────────────────────────────────────────────────────────
EOF
