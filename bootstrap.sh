#!/usr/bin/env bash
set -euo pipefail

echo "[BOOTSTRAP] Starting project bootstrap..."

# -----------------------------
# 1. uv 설치 확인
# -----------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[BOOTSTRAP] uv not found. Installing uv..."

  # 공식 권장 설치 방식
  curl -LsSf https://astral.sh/uv/install.sh | sh

  echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
  echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc

else
  echo "[BOOTSTRAP] uv already installed."
fi

# -----------------------------
# 2. 가상환경 생성
# -----------------------------
if [[ ! -d ".venv" ]]; then
  echo "[BOOTSTRAP] Creating virtual environment (.venv)..."
  uv venv .venv
else
  echo "[BOOTSTRAP] .venv already exists. Skipping."
fi

# -----------------------------
# 3. 의존성 설치
# -----------------------------
echo "[BOOTSTRAP] Installing Python dependencies..."
uv sync --all-groups

# -----------------------------
# 4. 환경 변수 설정
# -----------------------------

ENV_FILE=".env"
LAB_PRJ_ROOT_VALUE="$(pwd)"

if [[ -f "$ENV_FILE" ]]; then
  if grep -q '^LAB_PRJ_ROOT=' "$ENV_FILE"; then
    # 기존 LAB_PRJ_ROOT만 교체
    sed -i.bak "s|^LAB_PRJ_ROOT=.*|LAB_PRJ_ROOT=${LAB_PRJ_ROOT_VALUE}|" "$ENV_FILE"
    rm -f "${ENV_FILE}.bak"
  else
    # .env는 있으나 LAB_PRJ_ROOT 없음 → 추가
    echo "LAB_PRJ_ROOT=${LAB_PRJ_ROOT_VALUE}" >> "$ENV_FILE"
  fi
else
  # .env 자체가 없음 → 새로 생성
  echo "LAB_PRJ_ROOT=${LAB_PRJ_ROOT_VALUE}" > "$ENV_FILE"
fi

echo "[BOOTSTRAP] LAB_PRJ_ROOT set to ${LAB_PRJ_ROOT_VALUE}"

# -----------------------------
# 5. 한국어 폰트 설치
# -----------------------------
echo "[BOOTSTRAP] Installing korean fonts..."

bash ./script/install_korean_fonts.sh

echo "[BOOTSTRAP] Bootstrap completed successfully."
