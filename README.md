# ⚠️ Windows OpenMP 修复说明

本分支针对 Windows 下 ComfyUI、PyTorch 与 `llama-cpp-python` 同进程运行时的
OpenMP Runtime 冲突进行了修复。典型错误如下：

```text
OMP: Error #15: Initializing libomp140.x86_64.dll,
but found libiomp5md.dll already initialized.
Fatal Python error: Aborted
```

## 本次修改要点

- 提供 Windows CUDA wheel 源码构建流程，并通过 `-DGGML_OPENMP=OFF`
  禁用 GGML OpenMP。
- 修正 JamePeng fork 的打包逻辑，避免在 OpenMP 已禁用时仍将
  `libomp140.x86_64.dll` 放入 wheel。
- 构建完成后自动检查 wheel；如果仍携带或依赖 `libomp`、`libiomp` 或
  `vcomp`，校验将直接失败。
- 插件启动时提前检测 PyTorch 的 `libiomp5md.dll` 与 llama.cpp 的
  `libomp140.x86_64.dll`，同时存在时输出明确警告。
- 同时兼容新版 `mmproj_path` 和旧版 `clip_model_path`，修复 mmproj
  已成功加载却被误判为未配置的问题。

## 使用注意事项

> **重要：当前 `requirements.txt` 中的 Windows wheel 仍是原始
> `llama-cpp-python 0.3.40` 构建，并不包含本次 OpenMP 修复。**
> 在安全 wheel 发布到 GitHub Releases 并更新下载地址之前，请按照
> [Windows OpenMP-safe 构建指南](docs/windows-openmp.md)自行构建和安装。

- 不要使用 `KMP_DUPLICATE_LIB_OK=TRUE` 作为长期解决方案，并从 ComfyUI
  启动脚本及系统环境变量中移除它。
- wheel 必须与 ComfyUI 使用相同的 Python 次版本匹配，例如 Python 3.13
  必须安装 `cp313` wheel。
- `-DCMAKE_CUDA_ARCHITECTURES=120` 面向 RTX 5090/Blackwell；其他显卡请按
  对应 Compute Capability 重新构建。
- 安装 wheel 前请完全关闭 ComfyUI，然后使用 Portable 自带的
  `python_embeded\python.exe` 执行安装。
- ComfyUI Manager 更新或重新安装依赖后，可能会把安全 wheel 覆盖为原始
  wheel；更新后请重新检查 `llama_cpp\lib` 中是否出现
  `libomp140.x86_64.dll`。
- 安装后应在不设置 `KMP_DUPLICATE_LIB_OK` 的条件下测试普通生图工作流、
  GGUF/mmproj 图片推理以及多次连续执行。

## English: Windows OpenMP fix

This branch fixes an OpenMP runtime conflict that can occur when ComfyUI,
PyTorch, and `llama-cpp-python` run in the same Windows process. A typical
failure looks like this:

```text
OMP: Error #15: Initializing libomp140.x86_64.dll,
but found libiomp5md.dll already initialized.
Fatal Python error: Aborted
```

### What changed

- Added a reproducible Windows CUDA wheel build that disables GGML OpenMP with
  `-DGGML_OPENMP=OFF`.
- Fixed the JamePeng fork packaging logic so that
  `libomp140.x86_64.dll` is not included when OpenMP is disabled.
- Added post-build wheel validation. Validation fails if the wheel still
  bundles or imports `libomp`, `libiomp`, or `vcomp`.
- Added an early startup check for PyTorch's `libiomp5md.dll` and llama.cpp's
  `libomp140.x86_64.dll`, with an explicit warning when both are installed.
- Added compatibility for both the new `mmproj_path` property and the legacy
  `clip_model_path` property, preventing a successfully loaded mmproj from
  being reported as missing.

### Important notes

> **The Windows wheels currently referenced by `requirements.txt` are still
> the original `llama-cpp-python 0.3.40` builds and do not include this OpenMP
> fix.** Until an OpenMP-safe wheel is published in GitHub Releases and the
> download URL is updated, build and install one by following the
> [Windows OpenMP-safe build guide](docs/windows-openmp.md).

- Do not use `KMP_DUPLICATE_LIB_OK=TRUE` as a permanent solution. Remove it
  from ComfyUI startup scripts and Windows environment variables.
- The wheel must match ComfyUI's Python minor version. For example, Python
  3.13 requires a `cp313` wheel.
- `-DCMAKE_CUDA_ARCHITECTURES=120` targets RTX 5090/Blackwell. Rebuild with the
  appropriate Compute Capability for other GPUs.
- Completely close ComfyUI before installing the wheel, and run pip through
  Portable's own `python_embeded\python.exe`.
- ComfyUI Manager dependency updates may replace the safe wheel with the
  original build. After an update, check whether
  `llama_cpp\lib\libomp140.x86_64.dll` has returned.
- Validate without `KMP_DUPLICATE_LIB_OK`: test a normal image-generation
  workflow, GGUF/mmproj image inference, and several repeated executions.

---

# ComfyUI-llama-cpp  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]** 

## Preview  
![](./img/preview.jpg)

## 安装方法（Windows / RTX 50 系）

以下方法会从本仓库安装节点，并在本机编译不携带 LLVM OpenMP Runtime 的
`llama-cpp-python` CUDA wheel。请不要在 Windows 上直接执行原来的
`pip install -r requirements.txt`，其中引用的旧 Windows wheel 尚未包含
OpenMP 修复。

### 1. 克隆本仓库

```powershell
$portable = "C:\path\to\ComfyUI_windows_portable"
$python = Join-Path $portable "python_embeded\python.exe"

Set-Location (Join-Path $portable "ComfyUI\custom_nodes")
git clone https://github.com/yaocoler/ComfyUI-llama-cpp_vlmx.git
Set-Location .\ComfyUI-llama-cpp_vlmx
```

### 2. 安装 Python 基础依赖

```powershell
& $python -m pip install diskcache scipy numpy pillow gguf tqdm
```

### 3. 构建 OpenMP-safe CUDA wheel

需要提前安装 Visual Studio 2022 C++ Build Tools、Windows 11 SDK、Git、
CMake，以及支持目标显卡的 CUDA Toolkit。RTX 5090 / CUDA 12.8 示例：

```powershell
.\tools\build_windows_openmp_safe.ps1 `
  -PythonExe $python `
  -CudaArchitectures 120
```

脚本会自动应用源码补丁，以 `GGML_OPENMP=OFF` 构建，并拒绝任何仍携带或
依赖 `libomp`、`libiomp` 或 `vcomp` 的 wheel。

### 4. 安装构建结果

完全关闭 ComfyUI，然后执行：

```powershell
$wheel = Get-ChildItem .\dist\llama_cpp_python-*.whl |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

& $python -m pip install --force-reinstall --no-deps $wheel.FullName
```

从 ComfyUI 启动脚本和 Windows 环境变量中移除
`KMP_DUPLICATE_LIB_OK=TRUE`，再启动 ComfyUI。

### 5. 放置模型

- 将 GGUF 模型和对应的 mmproj 文件放入 `ComfyUI\models\LLM`。
- 在模型加载节点中同时选择 GGUF、匹配的 mmproj 和正确的 Chat Handler。
- 在不设置 `KMP_DUPLICATE_LIB_OK` 的情况下连续运行多次图片工作流。

更完整的编译和验收说明参见
[Windows OpenMP-safe 构建指南](docs/windows-openmp.md)。

## Installation (Windows / RTX 50 series)

This method installs the node from this repository and locally builds a CUDA
`llama-cpp-python` wheel without the LLVM OpenMP runtime. Do not use the old
`pip install -r requirements.txt` command on Windows yet: its referenced
Windows wheels do not contain this OpenMP fix.

### 1. Clone this repository

```powershell
$portable = "C:\path\to\ComfyUI_windows_portable"
$python = Join-Path $portable "python_embeded\python.exe"

Set-Location (Join-Path $portable "ComfyUI\custom_nodes")
git clone https://github.com/yaocoler/ComfyUI-llama-cpp_vlmx.git
Set-Location .\ComfyUI-llama-cpp_vlmx
```

### 2. Install the Python dependencies

```powershell
& $python -m pip install diskcache scipy numpy pillow gguf tqdm
```

### 3. Build the OpenMP-safe CUDA wheel

Install Visual Studio 2022 C++ Build Tools, the Windows 11 SDK, Git, CMake, and
a CUDA Toolkit supporting the target GPU first. For RTX 5090 with CUDA 12.8:

```powershell
.\tools\build_windows_openmp_safe.ps1 `
  -PythonExe $python `
  -CudaArchitectures 120
```

The script applies the source patch, builds with `GGML_OPENMP=OFF`, and rejects
any wheel that still bundles or imports `libomp`, `libiomp`, or `vcomp`.

### 4. Install the wheel

Completely close ComfyUI, then run:

```powershell
$wheel = Get-ChildItem .\dist\llama_cpp_python-*.whl |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

& $python -m pip install --force-reinstall --no-deps $wheel.FullName
```

Remove `KMP_DUPLICATE_LIB_OK=TRUE` from ComfyUI startup scripts and Windows
environment variables before starting ComfyUI.

### 5. Add models

- Put the GGUF model and matching mmproj file in `ComfyUI\models\LLM`.
- Select the GGUF, matching mmproj, and correct Chat Handler in the loader.
- Run the image workflow repeatedly without `KMP_DUPLICATE_LIB_OK`.

See the [Windows OpenMP-safe build guide](docs/windows-openmp.md) for complete
build and acceptance checks.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
