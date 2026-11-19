# Copyright 2025 The MiniMax team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MiniMax-M2 model."""

from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, load_balancing_loss_func
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import OutputRecorder, can_return_tuple
from ..llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, eager_attention_forward, rotate_half
from ..mistral.modeling_mistral import MistralAttention
from ..mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
)


class MiniMaxM2Config(PreTrainedConfig):
    r"""Configuration class for [`MiniMaxM2Model`] and [`MiniMaxM2ForCausalLM`].

    The configuration stores the dense attention and sparse mixture-of-experts parameters used by MiniMax-M2. The
    `rotary_dim` and `rope_theta` arguments are accepted for compatibility with original MiniMax checkpoints, including
    [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5).
    """

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1e6
    attribute_map = {"num_experts": "num_local_experts"}
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 14336,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_theta: float | None = 1e6,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        rotary_dim: int | None = None,
        sliding_window: int | None = None,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 2,
        num_local_experts: int | None = 8,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        router_jitter_noise: float | None = 0.0,
        use_qk_norm: bool | None = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads if head_dim is None else head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.use_qk_norm = use_qk_norm

        partial_rotary_factor = kwargs.pop("partial_rotary_factor", None)
        if rotary_dim is None:
            if partial_rotary_factor is None and isinstance(rope_parameters, dict):
                partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
            if partial_rotary_factor is None:
                partial_rotary_factor = 1.0
            rotary_dim = int(self.head_dim * partial_rotary_factor)
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = rotary_dim / self.head_dim if self.head_dim else 0.0
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            **kwargs,
        )


class MiniMaxM2Experts(MixtralExperts):
    pass


class MiniMaxM2TopKRouter(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = nn.functional.linear(hidden_states, self.weight)
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return router_logits, top_k_weights.to(router_logits.dtype), top_k_index


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.jitter_noise = config.router_jitter_noise
        self.gate = MiniMaxM2TopKRouter(config)
        self.experts = MiniMaxM2Experts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, selected_experts, routing_weights.to(hidden_states.dtype))
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class MiniMaxM2RMSNorm(LlamaRMSNorm):
    pass


class MiniMaxM2RotaryEmbedding(LlamaRotaryEmbedding):
    @staticmethod
    def compute_default_rope_parameters(
        config: MiniMaxM2Config | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = config.rotary_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat(((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1)
    k_embed = torch.cat(((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1)
    return q_embed, k_embed


class MiniMaxM2Attention(MistralAttention):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_attention_heads, eps=config.rms_norm_eps)
            self.k_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_key_value_heads, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class MiniMaxM2DecoderLayer(MixtralDecoderLayer):
    pass


class MiniMaxM2PreTrainedModel(MixtralPreTrainedModel):
    _no_split_modules = ["MiniMaxM2DecoderLayer"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(MiniMaxM2TopKRouter, layer_name="mlp.gate", index=0),
        "hidden_states": MiniMaxM2DecoderLayer,
        "attentions": MiniMaxM2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MiniMaxM2TopKRouter):
            init.zeros_(module.e_score_correction_bias)


class MiniMaxM2Model(MixtralModel):
    pass


class MiniMaxM2ForCausalLM(MixtralForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the causal language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniMaxM2ForCausalLM

        >>> model = MiniMaxM2ForCausalLM.from_pretrained("MiniMaxAI/MiniMax-M2.5")
        >>> tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2.5")
        >>> inputs = tokenizer("Hello, my name is", return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=20)
        >>> tokenizer.decode(outputs[0], skip_special_tokens=True)
        ```
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_attention_mask = attention_mask
            if attention_mask is not None and attention_mask.shape[1] != sequence_length:
                aux_attention_mask = attention_mask[:, -sequence_length:]
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                aux_attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["MiniMaxM2Config", "MiniMaxM2ForCausalLM", "MiniMaxM2Model", "MiniMaxM2PreTrainedModel"]
