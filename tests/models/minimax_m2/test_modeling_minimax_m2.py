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

import tempfile
import unittest

from transformers import AutoConfig, is_torch_available
from transformers.testing_utils import (
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import AutoModelForCausalLM, AutoTokenizer, MiniMaxM2Config, MiniMaxM2ForCausalLM, MiniMaxM2Model
    from transformers.core_model_loading import revert_weight_conversion
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention, MiniMaxM2TopKRouter

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MiniMaxM2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniMaxM2Model

    def __init__(self, parent):
        super().__init__(parent, batch_size=3, num_experts=4)
        self.hidden_act = "silu"


@require_torch
class MiniMaxM2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniMaxM2ModelTester
    # Not every expert or the routing-correction bias receives a gradient in a small routed batch.
    test_all_params_have_gradient = False

    def test_router_logits_recording(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_router_logits = True
        model = MiniMaxM2ForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertEqual(len(outputs.router_logits), config.num_hidden_layers)
        self.assertEqual(outputs.router_logits[0].shape, (inputs["input_ids"].numel(), config.num_local_experts))

    def test_qk_norm_uses_flattened_projection(self):
        config = self._tiny_config()
        attention = MiniMaxM2Attention(config, layer_idx=0)
        hidden_states = torch.randn(2, 3, config.hidden_size)
        query = attention.q_proj(hidden_states)
        key = attention.k_proj(hidden_states)

        self.assertEqual(attention.q_norm.weight.numel(), config.num_attention_heads * config.head_dim)
        self.assertEqual(attention.k_norm.weight.numel(), config.num_key_value_heads * config.head_dim)
        for projection in ("q_proj", "k_proj", "v_proj"):
            self.assertEqual(config.base_model_tp_plan[f"layers.*.self_attn.{projection}"], "colwise_rep")
        self.assertEqual(config.base_model_tp_plan["layers.*.self_attn.o_proj"], "rowwise_rep")
        torch.testing.assert_close(
            attention.q_norm(query),
            torch.nn.functional.rms_norm(
                query, (config.num_attention_heads * config.head_dim,), attention.q_norm.weight, config.rms_norm_eps
            ),
        )
        torch.testing.assert_close(
            attention.k_norm(key),
            torch.nn.functional.rms_norm(
                key, (config.num_key_value_heads * config.head_dim,), attention.k_norm.weight, config.rms_norm_eps
            ),
        )

    def test_router_uses_uncorrected_sigmoid_scores(self):
        config = self._tiny_config(num_experts_per_tok=2, num_local_experts=4)
        router = MiniMaxM2TopKRouter(config)
        with torch.no_grad():
            router.weight.zero_()
            router.weight[:, :4].copy_(torch.eye(4))
            router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0]))
        hidden_states = torch.zeros(1, config.hidden_size)
        hidden_states[0, :4] = torch.tensor([1.0, 0.5, -0.25, 0.75])

        logits, weights, selected = router(hidden_states)
        scores = logits.float().sigmoid()
        expected_selected = (scores + router.e_score_correction_bias).topk(2, dim=-1, sorted=False).indices
        expected_weights = scores.gather(1, expected_selected)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(selected, expected_selected)
        torch.testing.assert_close(weights, expected_weights.to(logits.dtype))
        self.assertEqual(selected[0].tolist(), [3, 0])

    def test_legacy_config(self):
        unused_options = {
            "attn_type_list": [1],
            "qk_norm_type": "per_layer",
            "scoring_func": "sigmoid",
            "use_qk_norm": True,
            "use_routing_bias": True,
            "shared_intermediate_size": 0,
            "use_mtp": True,
            "num_mtp_modules": 3,
            "mtp_transformer_layers": 1,
            "sliding_window": 128,
            "router_jitter_noise": 0.1,
        }
        config = AutoConfig.for_model(
            model_type="minimax_m2",
            hidden_size=16,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=2,
            num_experts_per_tok=1,
            vocab_size=48,
            rope_theta=5000000,
            rotary_dim=4,
            **unused_options,
        )
        self.assertIsInstance(config, MiniMaxM2Config)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.5)
        for name in (*unused_options, "partial_rotary_factor"):
            self.assertFalse(hasattr(config, name))

    def test_original_checkpoint_conversion_round_trip(self):
        config = self._tiny_config()
        model = MiniMaxM2ForCausalLM(config).eval()
        original_state = revert_weight_conversion(model, model.state_dict())
        self.assertIn("model.layers.0.block_sparse_moe.experts.0.w1.weight", original_state)
        self.assertIn("model.layers.0.block_sparse_moe.e_score_correction_bias", original_state)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            save_file(original_state, f"{tmpdir}/model.safetensors")
            loaded, loading_info = MiniMaxM2ForCausalLM.from_pretrained(
                tmpdir, attn_implementation="eager", output_loading_info=True
            )

        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)

    @staticmethod
    def _tiny_config(**overrides):
        kwargs = {
            "vocab_size": 48,
            "hidden_size": 16,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "partial_rotary_factor": 0.5,
            "pad_token_id": 0,
        }
        kwargs.update(overrides)
        return MiniMaxM2Config(**kwargs)


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
