#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Installing Korean fonts for visualization (Debian/Ubuntu)..."

# --- Debian 계열 체크 ---
if [[ ! -f /etc/debian_version ]]; then
  echo "[INFO] Non-Debian system detected. Skipping font installation."
  exit 0
fi

# --- sudo / root 자동 처리 ---
if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "[WARN] sudo not available. Skipping font installation."
    exit 0
  fi
fi

FAMILY_REGEX="Nanum"

print_font_family() {
  fc-list --format="%{family[0]}\n" | sort | uniq 2>/dev/null \
    | grep -iE "$FAMILY_REGEX" \
    | sed 's/^/- /' || true
}

# --- 이미 설치된 경우 스킵 ---
if command -v fc-list >/dev/null 2>&1; then
  if fc-list | grep -qiE "$FAMILY_REGEX"; then
    echo "[INFO] Korean fonts already installed. Skipping."
    echo "[INFO] Detected font family:"
    print_font_family
    exit 0
  fi
fi

# --- 설치 진행 ---
$SUDO apt-get update

$SUDO apt-get install -y --no-install-recommends \
  ca-certificates \
  fontconfig \
  fonts-nanum

# --- 캐시 갱신 ---
fc-cache -f

echo "[INFO] Installed fonts:"
print_font_family
