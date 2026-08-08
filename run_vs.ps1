# run_vs.ps1 <args...> - run a command inside the VS2022 x64 dev environment.
param([Parameter(ValueFromRemainingArguments=$true)] [string[]]$CommandArgs)
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if ($CommandArgs.Count -eq 0) { throw "usage: run_vs.ps1 <command and args>" }
$joined = ($CommandArgs -join ' ')
$cmd = "`"$vcvars`" >nul 2>&1 && $joined"
cmd /c $cmd
exit $LASTEXITCODE
