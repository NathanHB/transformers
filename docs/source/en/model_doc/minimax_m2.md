<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MiniMax-M2

## Overview

MiniMax-M2 is a sparse mixture-of-experts causal language model developed by MiniMax. The architecture combines grouped-query attention with partial rotary position embeddings and per-projection query/key normalization. Each decoder layer routes tokens to a subset of feed-forward experts using sigmoid routing scores and a correction bias for expert selection.

The native implementation supports the original MiniMax-M2.5 checkpoints, including their unpacked expert tensor format and block-scaled FP8 weights. The `quantization_config.dequantize` option can be used to dequantize FP8 checkpoints for devices without FP8 support.

```python
import torch
from transformers import AutoTokenizer, MiniMaxM2ForCausalLM

model_id = "MiniMaxAI/MiniMax-M2.5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = MiniMaxM2ForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Explain sparse mixture-of-experts models in one paragraph."}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=128)
print(tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True))
```

## MiniMaxM2Config

[[autodoc]] MiniMaxM2Config

## MiniMaxM2Model

[[autodoc]] MiniMaxM2Model
    - forward

## MiniMaxM2ForCausalLM

[[autodoc]] MiniMaxM2ForCausalLM
    - forward
