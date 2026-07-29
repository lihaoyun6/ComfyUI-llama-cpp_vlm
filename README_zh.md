# ComfyUI-llama-cpp
在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型。  
**[[📃English](./README.md)]**   

## 预览
![](./img/preview.jpg) 

## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

### 模型路径:
- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中.  

	> 在使用VLM模型进行图像推理之前, 请确保已经下载并选择了主模型对应的`mmproj`权重文件.

## 文生图提示词节点

`Text to Image Prompt` 可以将一个词语或词组扩写为文生图提示词。

```text
Llama-cpp Model Loader → Text to Image Prompt → CLIP Text Encode
```

在 `subject` 中输入主体（例如 `雨夜里的白猫`），需要时在
`setting_words` 中填写风格词。使用纯文本模型时，将 `mmproj` 和
`chat_handler` 设置为 `None`。

## 更新日志

### 2026-07-29

- 新增 `Text to Image Prompt` 文生图提示词节点。
- 改进 `mmproj` 兼容性。

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
