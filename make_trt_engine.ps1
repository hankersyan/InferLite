# make_trt_engine.ps1 - Build the sample TensorRT engine + regenerate manifest.
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" +
            "C:\Tools\nvidia\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9\bin;" + $env:PATH
cd c:\Test\triton\inferlite
Write-Output "== Build engine =="
python tools\make_trt_model.py
Write-Output "== Regenerate manifest =="
python tools\make_manifest.py --repo models
