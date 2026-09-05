$ErrorActionPreference = "Stop"
$root = "C:\Tools\openvino\openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64"
$py = "C:\Apps\anaconda3\envs\b312gpu\python.exe"
$repo = "c:\Test\triton\inferlite"
$env:OPENVINO_LIB_PATHS = "$root\runtime\bin\intel64\Release"
$env:PATH = "$root\runtime\bin\intel64\Release;" + $env:PATH
$env:PYTHONPATH = "$root\python"
Set-Location $repo
& cmd /c "`"$root\setupvars.bat`" >nul 2>&1 && `"$py`" tools\make_dynamic_batch_model.py --out models\priority_batch_model --model-name priority_batch_model --priority-levels 3 --default-priority 2 --preserve-ordering"
