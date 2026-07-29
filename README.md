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

> **Python 3.13 / Windows x64：** `requirements.txt` 已指向实测通过的
> `llama-cpp-python 0.3.44` OpenMP-safe CUDA wheel，可直接安装。
> 其他 Python 次版本暂时仍使用上游 wheel；如果遇到 OpenMP 冲突，请按照
> [Windows OpenMP-safe 构建指南](docs/windows-openmp.md)自行构建。

- 不要使用 `KMP_DUPLICATE_LIB_OK=TRUE` 作为长期解决方案，并从 ComfyUI
  启动脚本及系统环境变量中移除它。
- wheel 必须与 ComfyUI 使用相同的 Python 次版本匹配，例如 Python 3.13
  必须安装 `cp313` wheel。发布流程会同时构建 Python 3.10–3.14。
- 默认构建覆盖 CUDA 架构 `75;80;86;89;120`，对应常见 RTX
  20/30/40/50 系显卡；也可以通过参数缩小目标架构以加快本地构建。
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

> **Python 3.13 / Windows x64:** `requirements.txt` now installs the tested
> `llama-cpp-python 0.3.44` OpenMP-safe CUDA wheel directly. Other Python
> versions temporarily retain the upstream wheels; build locally with the
> [Windows OpenMP-safe build guide](docs/windows-openmp.md) if they conflict.

- Do not use `KMP_DUPLICATE_LIB_OK=TRUE` as a permanent solution. Remove it
  from ComfyUI startup scripts and Windows environment variables.
- The wheel must match ComfyUI's Python minor version. For example, Python
  3.13 requires a `cp313` wheel. The release workflow builds Python 3.10–3.14.
- The default build covers CUDA architectures `75;80;86;89;120` for common
  RTX 20/30/40/50-series GPUs. The list can be narrowed for faster local builds.
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

## 安装方法

### 常规安装（Windows / Python 3.13）

与普通 ComfyUI 节点相同，无需替换或升级 Python：

```powershell
$portable = "C:\path\to\ComfyUI_windows_portable"
$python = Join-Path $portable "python_embeded\python.exe"

Set-Location (Join-Path $portable "ComfyUI\custom_nodes")
git clone https://github.com/yaocoler/ComfyUI-llama-cpp_vlmx.git
& $python -m pip install -r .\ComfyUI-llama-cpp_vlmx\requirements.txt
```

完全关闭并重新启动 ComfyUI。请勿设置
`KMP_DUPLICATE_LIB_OK=TRUE`。

### 从源码构建安全 wheel

以下方法会从本仓库安装节点，并在本机编译不携带 LLVM OpenMP Runtime 的
`llama-cpp-python` CUDA wheel，适合其他 Python 或 GPU 环境。

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
  -CudaArchitectures "75;80;86;89;120"
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

## Installation

### Standard install (Windows / Python 3.13)

Install it like a normal ComfyUI custom node; do not replace or upgrade
ComfyUI's Python:

```powershell
$portable = "C:\path\to\ComfyUI_windows_portable"
$python = Join-Path $portable "python_embeded\python.exe"

Set-Location (Join-Path $portable "ComfyUI\custom_nodes")
git clone https://github.com/yaocoler/ComfyUI-llama-cpp_vlmx.git
& $python -m pip install -r .\ComfyUI-llama-cpp_vlmx\requirements.txt
```

Completely stop and restart ComfyUI. Do not set
`KMP_DUPLICATE_LIB_OK=TRUE`.

### Build a safe wheel from source

This method installs the node from this repository and locally builds a CUDA
`llama-cpp-python` wheel without the LLVM OpenMP runtime for other Python or
GPU environments.

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
  -CudaArchitectures "75;80;86;89;120"
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

## Text to Image Prompt

`Text to Image Prompt` expands a word or short phrase into a directly usable
text-to-image description with the selected llama.cpp model.

### Inputs

- `llama_model`: Connect the output of `Llama-cpp Model Loader`.
- `subject`: The main word or phrase, for example `a white cat in a rainy night`.
- `setting_words`: Optional style, scene, composition, lighting, or mood constraints.
- `language`: Generate the prompt in Chinese or English.
- `detail_level`: Choose concise, detailed, or extremely detailed output.
- `seed`: Controls repeatable generation.
- `parameters` (optional): Connect `Llama-cpp Parameters` to override generation settings.

The `image_prompt` output is a `STRING` and can be connected directly to a
text-to-image text encoder such as `CLIP Text Encode (Prompt)`.

```text
Llama-cpp Model Loader
        │
        ▼
Text to Image Prompt
        │ image_prompt
        ▼
CLIP Text Encode (Prompt)
```

For text-only prompt generation, select `None` for both `mmproj` and
`chat_handler` in the model loader.

## Changelog

### 2026-07-29

- Added the `Text to Image Prompt` node.
- Improved multimodal-projector detection for chat handlers that expose either
  `mmproj_path` or `clip_model_path`.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
