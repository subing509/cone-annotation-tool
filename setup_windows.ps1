param([switch]$CUDA)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path '.venv-yolo11\Scripts\python.exe')) {
    py -3.12 -m venv .venv-yolo11
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 x64 first.' }
}
$python = Join-Path $PSScriptRoot '.venv-yolo11\Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
$index = if ($CUDA) { 'https://download.pytorch.org/whl/cu124' } else { 'https://download.pytorch.org/whl/cpu' }
& $python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url $index
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Output 'Ready. Configure best.pt, then run cone_project\start_tool.ps1.'
