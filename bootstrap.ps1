# bootstrap.ps1
$ErrorActionPreference = "Stop"

Write-Host "[BOOTSTRAP] Starting project bootstrap..."

# -----------------------------
# 1. uv 설치 확인
# -----------------------------
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "[BOOTSTRAP] uv not found. Installing uv..."

    # Windows 권장 방식: winget 사용 (가능한 경우)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        winget install --id=astral-sh.uv  -e
    }
    else {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    }

    # 설치 직후 현재 세션에서 uv 인식 재확인
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv installation attempted but uv is still not found in PATH. Restart PowerShell and retry."
    }

    Write-Host "[BOOTSTRAP] uv installed."
}
else {
    Write-Host "[BOOTSTRAP] uv already installed."
}

# -----------------------------
# 2. 가상환경 생성
# -----------------------------
if (-not (Test-Path ".\.venv")) {
    Write-Host "[BOOTSTRAP] Creating virtual environment (.venv)..."
    uv venv .venv
}
else {
    Write-Host "[BOOTSTRAP] .venv already exists. Skipping."
}

# -----------------------------
# 3. 의존성 설치
# -----------------------------
Write-Host "[BOOTSTRAP] Installing Python dependencies..."
uv sync --all-groups

# -----------------------------
# 4. 환경 변수(.env) 설정
# -----------------------------
$envFile = ".\.env"
$labPrjRootValue = (Get-Location).Path
$line = "LAB_PRJ_ROOT=$labPrjRootValue"

if (Test-Path $envFile) {
    $content = Get-Content $envFile -Raw

    if ($content -match '(^|\r?\n)LAB_PRJ_ROOT=') {
        # 기존 LAB_PRJ_ROOT만 교체
        $updated = ($content -split "(\r?\n)") | ForEach-Object {
            if ($_ -match '^LAB_PRJ_ROOT=') { $line } else { $_ }
        }
        # 원본의 개행 스타일 보존 목적: 기본 Join 사용
        Set-Content -Path $envFile -Value ($updated -join "`r`n") -NoNewline
        Add-Content -Path $envFile -Value "`r`n"
    }
    else {
        # .env는 있으나 LAB_PRJ_ROOT 없음 → 추가
        Add-Content -Path $envFile -Value $line
    }
}
else {
    # .env 자체가 없음 → 새로 생성
    Set-Content -Path $envFile -Value $line
}

Write-Host "[BOOTSTRAP] LAB_PRJ_ROOT set to $labPrjRootValue"

Write-Host "[BOOTSTRAP] Bootstrap completed successfully."
