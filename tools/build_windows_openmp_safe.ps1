[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [string]$CudaArchitectures = "120",
    [string]$SourceCommit = "bab30611b4035bd69765d4856f907c763a6a69fb"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$source = Join-Path $root ".build\llama-cpp-python"
$dist = Join-Path $root "dist"
$patch = Join-Path $root "patches\llama-cpp-python-openmp-safe.patch"

if (-not (Test-Path -LiteralPath $source)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $source) | Out-Null
    git -c core.longpaths=true clone --recurse-submodules `
        https://github.com/JamePeng/llama-cpp-python.git $source
}

git -C $source -c core.longpaths=true checkout $SourceCommit
git -C $source -c core.longpaths=true submodule update --init --recursive

git -C $source apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $source apply --check $patch
    git -C $source apply $patch
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vs) {
    throw "Visual Studio 2022 C++ x64 build tools were not found."
}

$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd.exe /d /s /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}

& $python -m pip install --upgrade ninja scikit-build-core
$env:CMAKE_GENERATOR = "Ninja"
$env:CUDACXX = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
$cmakeArgs = @(
    "-DGGML_CUDA=ON",
    "-DGGML_OPENMP=OFF",
    "-DGGML_NATIVE=OFF",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
) -join ";"

New-Item -ItemType Directory -Force -Path $dist | Out-Null
& $python -m pip wheel --no-cache-dir --no-deps --no-build-isolation `
    --wheel-dir $dist --config-settings "cmake.args=$cmakeArgs" $source
if ($LASTEXITCODE -ne 0) {
    throw "Wheel build failed."
}

$wheel = Get-ChildItem $dist -Filter "llama_cpp_python-*.whl" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $python (Join-Path $PSScriptRoot "verify_openmp_wheel.py") $wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Wheel verification failed."
}
Write-Host "OpenMP-safe wheel: $($wheel.FullName)"
