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
from transformers.testing_utils import require_read_token, require_torch, require_torch_accelerator, slow


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
    # Not every expert or the routing-correction bias receives a gradient in a small routed batch.
    test_all_params_have_gradient = False


@require_torch_accelerator
@slow
@require_read_token
class MiniMaxM2IntegrationTest(unittest.TestCase):
    def test_released_checkpoint_generation(self):
        checkpoint = "MiniMaxAI/MiniMax-M2.5"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        self.assertEqual(generated[0, -1:].tolist(), [8261])
        response = tokenizer.decode(generated[0, -1:], skip_special_tokens=True)
        self.assertEqual(response, " Paris")


if __name__ == "__main__":
    unittest.main()
