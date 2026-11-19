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

*This model was released on 2025-10-27 and added to Hugging Face Transformers on 2026-11-18.*

## Overview

[MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) is a sparse mixture-of-experts causal language model developed by MiniMax and introduced in the [official announcement](https://www.minimax.io/news/minimax-m2). The architecture combines grouped-query attention with partial rotary position embeddings and per-projection query/key normalization. Each decoder layer routes tokens to a subset of feed-forward experts using sigmoid routing scores and a correction bias for expert selection. The released checkpoints, including MiniMax-M2.5, are available in the [MiniMax-M2 collection](https://huggingface.co/collections/MiniMaxAI/minimax-m2-69607bef9bd3881ee76e11cb).

```python
import torch
from transformers import AutoTokenizer, MiniMaxM2ForCausalLM

model_id = "MiniMaxAI/MiniMax-M2.5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = MiniMaxM2ForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Explain sparse mixture-of-experts models in one paragraph."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## MiniMaxM2Config

[[autodoc]] MiniMaxM2Config

## MiniMaxM2Model

[[autodoc]] MiniMaxM2Model
    - forward

## MiniMaxM2ForCausalLM

[[autodoc]] MiniMaxM2ForCausalLM
    - forward
