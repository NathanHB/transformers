# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch MiniMax-M2 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_large_accelerator,
    require_torch_multi_accelerator,
    slow,
)


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer, MiniMaxM2Model

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MiniMaxM2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniMaxM2Model


@require_torch
class MiniMaxM2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniMaxM2ModelTester


# The released FP8 checkpoint needs multiple large accelerators for its roughly 230B parameters.
@require_torch
@require_torch_multi_accelerator
@require_torch_large_accelerator(memory=64)
@slow
class MiniMaxM2IntegrationTest(unittest.TestCase):
    def test_released_checkpoint_generation(self):
        checkpoint = "MiniMaxAI/MiniMax-M2.5"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", dtype="auto")
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_with_cache = model.generate(
                **inputs, max_new_tokens=20, min_new_tokens=20, do_sample=False, use_cache=True
            )
            generated_without_cache = model.generate(
                **inputs, max_new_tokens=20, min_new_tokens=20, do_sample=False, use_cache=False
            )

        self.assertEqual(generated_with_cache.shape[1], inputs.input_ids.shape[1] + 20)
        self.assertEqual(generated_with_cache.tolist(), generated_without_cache.tolist())


if __name__ == "__main__":
    unittest.main()
