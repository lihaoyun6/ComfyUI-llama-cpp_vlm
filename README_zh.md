# ComfyUI-llama-cpp
在ComfyUI中基于llama.cpp原生运行LLM/VLM模型。  
**[[📃English](./README.md)]**

## 更新日志
#### 2025-11-03
- 首次上传, 支持Qwen3-VL      

## 预览
![](./img/preview.jpg) 

## 安装步骤

#### 安装节点:
```bash
# Windows/Linux
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

```bash
# for macOS (Apple Silicon)
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt

# 使用 --no-binary 确保进行源码编译，--force-reinstall 用于覆盖任何现有的纯 CPU 版本，DGGML_METAL 则用于为 Apple Silicon 启用 Metal 加速
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install "git+https://github.com/JamePeng/llama-cpp-python.git" --force-reinstall --no-binary llama-cpp-python
```

> 在使用VLM模型处理图像之前, 请确保已经下载并选择了对应的`mmproj`权重.

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
