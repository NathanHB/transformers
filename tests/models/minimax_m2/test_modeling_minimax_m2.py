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
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM, MiniMaxM2Model
    from transformers.core_model_loading import revert_weight_conversion
    from transformers.models.minimax_m2.modeling_minimax_m2 import (
        MiniMaxM2Attention,
        MiniMaxM2TopKRouter,
        apply_rotary_pos_emb,
        rotate_half,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MiniMaxM2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniMaxM2Model

    def __init__(self, parent):
        super().__init__(parent, batch_size=3, num_experts=4)
        self.hidden_act = "silu"
        self.rotary_dim = self.head_dim // 2
        self.use_qk_norm = True


@require_torch
class MiniMaxM2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniMaxM2ModelTester
    test_all_params_have_gradient = False

    # There is no tiny hub checkpoint for this architecture.
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def test_load_balancing_loss(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_router_logits = True
        model = MiniMaxM2ForCausalLM(config).to(torch_device).eval()
        outputs = model(**inputs)
        self.assertEqual(len(outputs.router_logits), config.num_hidden_layers)
        self.assertEqual(outputs.router_logits[0].shape[0], inputs["input_ids"].numel())
        self.assertEqual(outputs.router_logits[0].shape[1], config.num_local_experts)
        self.assertTrue(torch.isfinite(outputs.aux_loss))

    def test_qk_norm_uses_flattened_projection(self):
        config = self._tiny_config()
        attention = MiniMaxM2Attention(config, layer_idx=0)
        hidden_states = torch.randn(2, 3, config.hidden_size)
        query = attention.q_proj(hidden_states)
        key = attention.k_proj(hidden_states)

        self.assertEqual(attention.q_norm.weight.numel(), config.num_attention_heads * config.head_dim)
        self.assertEqual(attention.k_norm.weight.numel(), config.num_key_value_heads * config.head_dim)
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

    def test_partial_rotary_embedding(self):
        query = torch.randn(2, 2, 3, 8)
        key = torch.randn(2, 2, 3, 8)
        cos = torch.randn(2, 3, 4)
        sin = torch.randn(2, 3, 4)
        rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        expected_query = torch.cat((query[..., :4] * cos + rotate_half(query[..., :4]) * sin, query[..., 4:]), dim=-1)
        expected_key = torch.cat((key[..., :4] * cos + rotate_half(key[..., :4]) * sin, key[..., 4:]), dim=-1)
        torch.testing.assert_close(rotated_query, expected_query)
        torch.testing.assert_close(rotated_key, expected_key)

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

    def test_outputs_cache_and_logits_to_keep(self):
        config = self._tiny_config(num_hidden_layers=2)
        config.output_router_logits = True
        config._attn_implementation = "eager"
        model = MiniMaxM2ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], device=torch_device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                output_router_logits=True,
                logits_to_keep=2,
            )

        self.assertEqual(outputs.logits.shape, (2, 2, config.vocab_size))
        self.assertEqual(len(outputs.hidden_states), 3)
        self.assertEqual(len(outputs.attentions), 2)
        self.assertEqual(len(outputs.router_logits), 2)
        self.assertEqual(outputs.past_key_values.get_seq_length(), 5)
        self.assertIsNotNone(outputs.aux_loss)

        next_token = torch.randint(0, config.vocab_size, (2, 1), device=torch_device)
        next_mask = torch.ones(2, 6, dtype=torch.long, device=torch_device)
        with torch.no_grad():
            next_outputs = model(next_token, attention_mask=next_mask, past_key_values=outputs.past_key_values)
        self.assertEqual(next_outputs.past_key_values.get_seq_length(), 6)
        self.assertEqual(len(next_outputs.router_logits), 2)
        self.assertIsNotNone(next_outputs.aux_loss)

    def test_legacy_config_and_meta_construction(self):
        legacy_metadata = {
            "attn_type_list": [1],
            "qk_norm_type": "per_layer",
            "scoring_func": "sigmoid",
            "use_routing_bias": True,
            "shared_intermediate_size": 0,
            "use_mtp": True,
            "num_mtp_modules": 3,
            "mtp_transformer_layers": 1,
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
            **legacy_metadata,
        )
        self.assertIsInstance(config, MiniMaxM2Config)
        self.assertEqual(config.partial_rotary_factor, 0.5)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.5)
        for name, value in legacy_metadata.items():
            self.assertEqual(getattr(config, name), value)
        with torch.device("meta"):
            model = MiniMaxM2ForCausalLM(config)
        self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))

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

    def test_fp8_checkpoint_dequantization_and_expert_packing(self):
        config = self._tiny_config()
        model = MiniMaxM2ForCausalLM(config)
        original_state = revert_weight_conversion(model, model.state_dict())
        expected = {}
        quantized_state = {}
        for name, tensor in original_state.items():
            if self._is_fp8_weight(name):
                quantized, scales, dequantized = self._quantize_fp8(tensor)
                quantized_state[name] = quantized
                quantized_state[f"{name}_scale_inv"] = scales
                expected[name] = dequantized
            else:
                quantized_state[name] = tensor

        config.quantization_config = {
            "quant_method": "fp8",
            "fmt": "float8_e4m3fn",
            "activation_scheme": "dynamic",
            "weight_block_size": [8, 8],
            "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head"],
        }
        packed_state = dict(model.state_dict())
        prefix = "model.layers.0.mlp.experts"
        packed_state[f"{prefix}.gate_up_proj"] = torch.stack(
            [
                torch.cat(
                    (
                        quantized_state[f"model.layers.0.block_sparse_moe.experts.{expert}.w1.weight"],
                        quantized_state[f"model.layers.0.block_sparse_moe.experts.{expert}.w3.weight"],
                    )
                )
                for expert in range(config.num_local_experts)
            ]
        )
        packed_state[f"{prefix}.gate_up_proj_scale_inv"] = torch.stack(
            [
                torch.cat(
                    (
                        quantized_state[f"model.layers.0.block_sparse_moe.experts.{expert}.w1.weight_scale_inv"],
                        quantized_state[f"model.layers.0.block_sparse_moe.experts.{expert}.w3.weight_scale_inv"],
                    )
                )
                for expert in range(config.num_local_experts)
            ]
        )
        packed_state[f"{prefix}.down_proj"] = torch.stack(
            [
                quantized_state[f"model.layers.0.block_sparse_moe.experts.{expert}.w2.weight"]
                for expert in range(config.num_local_experts)
            ]
        )
        packed_state[f"{prefix}.down_proj_scale_inv"] = torch.stack(
            [
                quantized_state[f"model.layers.0.block_sparse_moe.experts.{expert}.w2.weight_scale_inv"]
                for expert in range(config.num_local_experts)
            ]
        )
        reverted_state = revert_weight_conversion(model, packed_state)
        for name, tensor in quantized_state.items():
            if ".block_sparse_moe.experts." in name:
                torch.testing.assert_close(reverted_state[name], tensor, rtol=0, atol=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            save_file(quantized_state, f"{tmpdir}/model.safetensors")
            loaded, loading_info = MiniMaxM2ForCausalLM.from_pretrained(
                tmpdir, attn_implementation="eager", output_loading_info=True
            )

        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        self.assertFalse(loaded.is_quantized)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"model.layers.0.self_attn.{projection}.weight"
            actual = getattr(loaded.model.layers[0].self_attn, projection).weight
            torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)

        gate_up = torch.stack(
            [
                torch.cat(
                    (
                        expected[f"model.layers.0.block_sparse_moe.experts.{expert}.w1.weight"],
                        expected[f"model.layers.0.block_sparse_moe.experts.{expert}.w3.weight"],
                    )
                )
                for expert in range(config.num_local_experts)
            ]
        )
        down = torch.stack(
            [
                expected[f"model.layers.0.block_sparse_moe.experts.{expert}.w2.weight"]
                for expert in range(config.num_local_experts)
            ]
        )
        torch.testing.assert_close(loaded.model.layers[0].mlp.experts.gate_up_proj, gate_up, rtol=0, atol=0)
        torch.testing.assert_close(loaded.model.layers[0].mlp.experts.down_proj, down, rtol=0, atol=0)

        input_ids = torch.randint(0, config.vocab_size, (2, 4))
        attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        with torch.no_grad():
            outputs = loaded(input_ids, attention_mask=attention_mask, output_router_logits=True)
        self.assertTrue(torch.isfinite(outputs.logits).all())

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
            "rotary_dim": 4,
            "use_qk_norm": True,
            "pad_token_id": 0,
        }
        kwargs.update(overrides)
        return MiniMaxM2Config(**kwargs)

    @staticmethod
    def _is_fp8_weight(name):
        return (
            ".self_attn." in name and name.endswith(".weight") and "norm" not in name
        ) or ".block_sparse_moe.experts." in name

    @staticmethod
    def _quantize_fp8(weight):
        scales = torch.rand(weight.shape[0] // 8, weight.shape[1] // 8) * 0.05 + 0.01
        expanded_scales = scales.repeat_interleave(8, dim=0).repeat_interleave(8, dim=1)
        quantized = (weight / expanded_scales).to(torch.float8_e4m3fn)
        dequantized = quantized.float() * expanded_scales
        return quantized, scales, dequantized


if __name__ == "__main__":
    unittest.main()
