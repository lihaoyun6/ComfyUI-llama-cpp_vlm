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

---

# ComfyUI-llama-cpp  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]** 

## Preview  
![](./img/preview.jpg)

## Installation  

#### Install the node:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

	> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

### Windows OpenMP safety

Some Windows `llama-cpp-python` wheels conflict with PyTorch's OpenMP runtime
and can abort ComfyUI during image decode. `KMP_DUPLICATE_LIB_OK` is not a
permanent fix. See the [OpenMP-safe CUDA build guide](docs/windows-openmp.md).

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
