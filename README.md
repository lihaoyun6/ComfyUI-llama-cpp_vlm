# ComfyUI-llama-cpp  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]**

## Changelog  
#### 2025-11-03  
- Initial release, added support for Qwen3-VL  

## Preview  
![](./img/preview.jpg)

## Installation  

#### Install the node:  
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

# We use --no-binary to ensure compilation and --force-reinstall to overwrite any CPU versions, DGGML_METAL to enable Metal acceleration for Apple Silicon
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install "git+https://github.com/JamePeng/llama-cpp-python.git" --force-reinstall --no-binary llama-cpp-python
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
