param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$datasetRoot = Split-Path -Parent $projectDir
$pythonExe = Join-Path $datasetRoot '.venv-yolo11\Scripts\python.exe'
$scriptFile = Join-Path $projectDir 'train_yolo11.py'
$pidFile = Join-Path $projectDir 'training_process.json'
if (Test-Path -LiteralPath $pidFile) {
    $previous = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) {
        throw "Training process $($previous.pid) is still running."
    }
}
$arguments = @('-u', ('"' + $scriptFile + '"'))
if ($Resume) { $arguments += '--resume' }
$process = Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $projectDir -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectDir 'training.stdout.log') -RedirectStandardError (Join-Path $projectDir 'training.stderr.log') -PassThru
@{pid=$process.Id; started_at=(Get-Date).ToString('o'); script=$scriptFile; resume=[bool]$Resume} | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
Write-Output "Training PID: $($process.Id)"

