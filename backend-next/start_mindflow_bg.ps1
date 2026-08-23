# Hidden backend launcher used by start_mindflow_bg.vbs.
# Starts python -m mindflow.main, writes a PID file, and probes the health
# endpoint so a missing autostart is visible in service-logs.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$serviceLogs = Join-Path $env:LOCALAPPDATA "mindflow\service-logs"
$runtimeLogs = Join-Path $root "runtime_logs"
New-Item -ItemType Directory -Force -Path $serviceLogs, $runtimeLogs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outLog = Join-Path $runtimeLogs "backend-$stamp.out.log"
$errLog = Join-Path $runtimeLogs "backend-$stamp.err.log"
$pidFile = Join-Path $serviceLogs "backend.pid"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    "python not found on PATH" | Out-File -FilePath $errLog -Encoding utf8
    exit 1
}

# If the backend is already healthy, do not spawn a duplicate process.
try {
    $existing = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8765/api/v1/health/live" -TimeoutSec 2
    if ($existing.StatusCode -eq 200) {
        "already running; health ok at $(Get-Date -Format o)" | Out-File -FilePath (Join-Path $serviceLogs "latest_backend_logs.txt") -Append -Encoding utf8
        exit 0
    }
} catch {
    # Not running yet - continue to start it.
}

# src-layout package: python -m mindflow.main needs src on PYTHONPATH.
$srcPath = Join-Path $root "src"
if ($env:PYTHONPATH -and $env:PYTHONPATH -notlike "*$srcPath*") {
    $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
} elseif (-not $env:PYTHONPATH) {
    $env:PYTHONPATH = $srcPath
}

# Force UTF-8 mode to prevent GBK encoding errors on Windows
$env:PYTHONUTF8 = "1"

$proc = Start-Process -FilePath $python.Source `
    -ArgumentList @("-m", "mindflow.main") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru

$proc.Id | Out-File -FilePath $pidFile -Encoding ascii
"started pid=$($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath (Join-Path $serviceLogs "latest_backend_logs.txt") -Encoding utf8

$deadline = (Get-Date).AddSeconds(90)
$ok = $false
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8765/api/v1/health/live" -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $ok = $true; break }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

"health=$ok at $(Get-Date -Format o)" | Out-File -FilePath (Join-Path $serviceLogs "latest_backend_logs.txt") -Append -Encoding utf8
if (-not $ok) {
    "backend did not become healthy within 90s; see $errLog" | Out-File -FilePath (Join-Path $serviceLogs "latest_backend_logs.txt") -Append -Encoding utf8
    exit 2
}
