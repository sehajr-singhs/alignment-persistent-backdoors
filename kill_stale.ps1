# Kill any stale `python -m backdoors.run_all` processes left over from a
# previous (possibly duplicated) chain launch.  chain2.sh calls this before
# starting, so two chains can never race on the same result files.
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'backdoors\.run_all' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Output "stale pipeline processes killed"
