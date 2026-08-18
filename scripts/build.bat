@echo off
REM Build InferLite with MSVC (VS2022) + CMake + Ninja.
REM Auto-detects the Visual Studio installation via vswhere.
setlocal

set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found at "%VSWHERE%"
    exit /b 1
)

for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (
    echo [ERROR] No Visual Studio with VC tools found.
    exit /b 1
)

set "VCVARS=%VS%\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

if not exist "%VCVARS%" (
    echo [ERROR] vcvars64.bat not found at "%VCVARS%"
    exit /b 1
)
if not exist "%CMAKE%" (
    echo [ERROR] cmake not found at "%CMAKE%"
    exit /b 1
)

call "%VCVARS%" >nul

REM Repo root is the parent of this scripts\ directory.
set "REPO=%~dp0.."

set "BUILDDIR=%REPO%\build"
if not exist "%BUILDDIR%" mkdir "%BUILDDIR%"

"%CMAKE%" -S "%REPO%" -B "%BUILDDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%NINJA%"
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%BUILDDIR%"
if errorlevel 1 exit /b 1

echo.
echo Build complete: %BUILDDIR%\inferlite.exe
endlocal
