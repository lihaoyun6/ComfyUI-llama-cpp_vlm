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
