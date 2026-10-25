param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$datasetRoot = Split-Path -Parent $projectDir
$pythonExe = Join-Path $datasetRoot '.venv-yolo11\Scripts\python.exe'
$scriptFile = Join-Path $projectDir 'tool_server.py'
$probe = [System.Net.Sockets.TcpClient]::new()
$portInUse = $false
try { $probe.Connect('127.0.0.1', $Port); $portInUse = $true } catch [System.Net.Sockets.SocketException] { } finally { $probe.Dispose() }
if ($portInUse) { throw "Port $Port is already in use. Open http://127.0.0.1:$Port or select a different port." }
$process = Start-Process -FilePath $pythonExe -ArgumentList @('-u',('"'+$scriptFile+'"'),'--port',$Port) -WorkingDirectory $projectDir -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectDir 'tool.stdout.log') -RedirectStandardError (Join-Path $projectDir 'tool.stderr.log') -PassThru
@{pid=$process.Id;port=$Port;started_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectDir 'tool_process.json') -Encoding utf8
Write-Output "Cone tool: http://127.0.0.1:$Port (PID $($process.Id))"
