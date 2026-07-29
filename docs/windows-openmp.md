# Windows OpenMP conflict

PyTorch loads Intel OpenMP from `torch\lib\libiomp5md.dll`. Some Windows
`llama-cpp-python` wheels also load LLVM OpenMP from
`llama_cpp\lib\libomp140.x86_64.dll`. Loading both runtimes in ComfyUI can
terminate Python with `OMP Error #15` during image decode.

`KMP_DUPLICATE_LIB_OK=TRUE` only suppresses the runtime guard and is not a
supported fix.

## Build an OpenMP-safe CUDA wheel

Requirements:

- Visual Studio 2022 C++ x64 build tools and Windows 11 SDK
- CMake and Git
- CUDA Toolkit 12.8 or newer for RTX 50-series GPUs
- The same Python minor version used by ComfyUI Portable

Run:

```powershell
.\tools\build_windows_openmp_safe.ps1 `
  -PythonExe C:\path\to\ComfyUI_windows_portable\python_embeded\python.exe `
  -CudaArchitectures 120
```

The build uses CUDA with `GGML_OPENMP=OFF` and verifies that the resulting
wheel neither bundles nor directly imports an OpenMP runtime.

Install it after closing ComfyUI:

```powershell
C:\path\to\python.exe -m pip install --force-reinstall --no-deps `
  .\dist\llama_cpp_python-*.whl
```

Remove `set KMP_DUPLICATE_LIB_OK=TRUE` from the ComfyUI startup batch file.
Then verify normal image workflows and repeated GGUF/mmproj image decoding.
