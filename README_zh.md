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

`Text to Image Prompt` 节点可以通过选定的 llama.cpp 模型，将一个词语或
词组扩写为可直接用于文生图的画面描述。

### 输入

- `llama_model`：连接 `Llama-cpp Model Loader` 的输出。
- `subject`：主体词语或词组，例如 `雨夜里的白猫`。
- `setting_words`：可选的风格、场景、构图、光线或氛围设定词。
- `language`：选择生成中文或英文提示词。
- `detail_level`：选择简洁、详细或极致详细。
- `seed`：控制可复现的生成结果。
- `parameters`（可选）：连接 `Llama-cpp Parameters`，覆盖默认生成参数。

节点输出 `image_prompt`，类型为 `STRING`，可以直接连接
`CLIP Text Encode (Prompt)` 等文生图文本编码节点。

```text
Llama-cpp Model Loader
        │
        ▼
Text to Image Prompt
        │ image_prompt
        ▼
CLIP Text Encode (Prompt)
```

仅使用文本模型生成提示词时，请在模型加载节点中将 `mmproj` 和
`chat_handler` 都设置为 `None`。

## 更新日志

### 2026-07-29

- 新增 `Text to Image Prompt` 文生图提示词节点。
- 改进多模态投影模块检测，同时兼容聊天处理器的 `mmproj_path` 和
  `clip_model_path` 属性。

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
