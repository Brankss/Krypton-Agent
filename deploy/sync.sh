#!/usr/bin/env bash
#
# Krypton Agent — GitHub auto-sync.
#
# Pulls the latest commit for whatever branch this checkout is on. If the
# branch moved, it reinstalls dependencies (only when pyproject.toml changed)
# and restarts the bot. Invoked by krypton-sync.timer; safe to run by hand.
#
# A restart mid-conversation is safe: Krypton persists each Telegram chat's
# context to SQLite and restores it on the next message.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git fetch --quiet origin "$BRANCH"

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"
[ "$LOCAL" = "$REMOTE" ] && exit 0          # already up to date — nothing to do

echo "krypton-sync: $BRANCH $LOCAL -> $REMOTE"
git reset --hard "origin/$BRANCH"           # keeps gitignored data/, inbox/, .env

if ! git diff --quiet "$LOCAL" "$REMOTE" -- pyproject.toml; then
  echo "krypton-sync: dependencies changed -> reinstalling"
  "$REPO/.venv/bin/pip" install -q -e "$REPO"
fi

sudo systemctl restart krypton.service
echo "krypton-sync: redeployed $REMOTE and restarted krypton.service"
