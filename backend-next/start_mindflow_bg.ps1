# Hidden backend launcher used by start_mindflow_bg.vbs.
# Starts python -m mindflow.main, writes a PID file, and probes the health
# endpoint so a missing autostart is visible in service-logs.
param([switch]$OpenUi)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$serviceLogs = Join-Path $env:LOCALAPPDATA "mindflow\service-logs"
$runtimeLogs = Join-Path $root "runtime_logs"
New-Item -ItemType Directory -Force -Path $serviceLogs, $runtimeLogs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outLog = Join-Path $runtimeLogs "backend-$stamp.out.log"
$errLog = Join-Path $runtimeLogs "backend-$stamp.err.log"
$pidFile = Join-Path $serviceLogs "backend.pid"

$uv = Get-Command uv -ErrorAction SilentlyContinue
$uvPath = if ($uv) { $uv.Source } else { Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
if (-not (Test-Path -LiteralPath $uvPath)) {
    "uv not found; install the project environment first" | Out-File -FilePath $errLog -Encoding utf8
    exit 1
}

function Open-MindFlowUi {
    if (-not $OpenUi) { return }
    Push-Location $root
    try {
        $uiUrl = & $uvPath run --locked --no-sync python -m mindflow.bootstrap
        if ($LASTEXITCODE -ne 0 -or $uiUrl -notmatch '^http://127\.0\.0\.1:8765/#bootstrap=') {
            throw "Unable to create the one-time MindFlow launch ticket"
        }
        $edge = Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"
        Start-Process -FilePath $edge -ArgumentList $uiUrl
    } finally {
        Pop-Location
    }
}

$env:PYTHONUTF8 = "1"

# If the backend is already healthy, do not spawn a duplicate process.
$alreadyRunning = $false
try {
    $existing = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/v1/health/ready" -TimeoutSec 3
    $alreadyRunning = $existing.status -eq "ready"
} catch {
    # Not running yet - continue to start it.
}
if ($alreadyRunning) {
    "already running; readiness ok at $(Get-Date -Format o)" | Out-File -FilePath (Join-Path $serviceLogs "latest_backend_logs.txt") -Append -Encoding utf8
    Open-MindFlowUi
    exit 0
}

$proc = Start-Process -FilePath $uvPath `
    -ArgumentList @("run", "--locked", "--no-sync", "python", "-m", "mindflow.main") `
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
    if ($proc.HasExited) { break }
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/v1/health/ready" -TimeoutSec 3
        if ($resp.status -eq "ready") { $ok = $true; break }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

"health=$ok at $(Get-Date -Format o)" | Out-File -FilePath (Join-Path $serviceLogs "latest_backend_logs.txt") -Append -Encoding utf8
if (-not $ok) {
    "backend did not become healthy within 90s; see $errLog" | Out-File -FilePath (Join-Path $serviceLogs "latest_backend_logs.txt") -Append -Encoding utf8
    exit 2
}
Open-MindFlowUi
