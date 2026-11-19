# Copyright 2026 The MiniMax team and The HuggingFace Inc. team. All rights reserved.
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
from torch.nn import init

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, load_balancing_loss_func
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import OutputRecorder, can_return_tuple
from ..glm.modeling_glm import GlmRotaryEmbedding
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from ..llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ..mistral.modeling_mistral import MistralAttention
from ..mixtral.configuration_mixtral import MixtralConfig
from ..mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralSparseMoeBlock,
    MixtralTopKRouter,
)


class MiniMaxM2Config(MixtralConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxM2Model`]. It is used to instantiate a
    MiniMax-M2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a configuration similar to that of
    [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 200064):
            Vocabulary size of the MiniMax-M2 model. Defines the number of different tokens that can be represented by
            the `input_ids` passed when calling [`MiniMaxM2Model`].
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimension of each expert's MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 62):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads used to implement grouped-query attention. If
            `num_key_value_heads=num_attention_heads`, the model uses multi-head attention; if
            `num_key_value_heads=1`, it uses multi-query attention. If set to `None`, this defaults to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head. If set to `None`, this defaults to
            `hidden_size // num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the experts.
        max_position_embeddings (`int`, *optional*, defaults to 196608):
            Maximum sequence length supported by the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation used to initialize weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return key/value caches for autoregressive decoding.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 200020):
            The id of the end-of-sequence token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the input and output word embeddings should be tied.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts selected for each token.
        num_local_experts (`int`, *optional*, defaults to 256):
            Number of experts in each sparse MLP layer.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to return router logits and compute the auxiliary load-balancing loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient of the auxiliary load-balancing loss in the total loss.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Fraction of each attention head dimension to which rotary embeddings are applied. The legacy `rotary_dim`
            argument is converted to this factor when loading original checkpoints.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Parameters for rotary position embeddings. By default, `rope_theta` is 5000000.0, `rope_type` is
            `"default"`, and `partial_rotary_factor` is 0.5. Legacy `rope_theta` and `rope_scaling` arguments are
            also accepted.
        kwargs (`dict`, *optional*):
            Additional configuration arguments, including metadata stored in original MiniMax checkpoints.

    ```python
    >>> from transformers import MiniMaxM2Config, MiniMaxM2Model

    >>> configuration = MiniMaxM2Config()
    >>> model = MiniMaxM2Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "minimax_m2"
    default_theta = 5_000_000.0

    def __init__(
        self,
        vocab_size: int | None = 200064,
        hidden_size: int | None = 3072,
        intermediate_size: int | None = 1536,
        num_hidden_layers: int | None = 62,
        num_attention_heads: int | None = 48,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 196608,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 200020,
        tie_word_embeddings: bool | None = False,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 8,
        num_local_experts: int | None = 256,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        partial_rotary_factor: float | None = None,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        **kwargs,
    ):
        kwargs.pop("use_qk_norm", None)  # Q/K normalization is always enabled for MiniMax-M2.
        kwargs.pop("router_jitter_noise", None)
        kwargs.pop("sliding_window", None)
        rotary_dim = kwargs.pop("rotary_dim", None)
        if rotary_dim is not None:
            rotary_head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
            partial_rotary_factor = rotary_dim / rotary_head_dim
        kwargs["partial_rotary_factor"] = 0.5 if partial_rotary_factor is None else partial_rotary_factor
        # The released architecture uses full attention and deterministic routing.
        sliding_window = None
        router_jitter_noise = 0.0

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            num_local_experts=num_local_experts,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            router_jitter_noise=router_jitter_noise,
            rope_parameters=rope_parameters,
            **kwargs,
        )
        self.partial_rotary_factor = self.rope_parameters["partial_rotary_factor"]


class MiniMaxM2Experts(MixtralExperts):
    pass


class MiniMaxM2TopKRouter(MixtralTopKRouter):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return router_logits, top_k_weights.to(router_logits.dtype), top_k_index


class MiniMaxM2SparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__(config)
        self.gate = MiniMaxM2TopKRouter(config)
        del self.jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, selected_experts, routing_weights.to(hidden_states.dtype))
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class MiniMaxM2RMSNorm(LlamaRMSNorm):
    pass


class MiniMaxM2RotaryEmbedding(GlmRotaryEmbedding):
    pass


class MiniMaxM2Attention(MistralAttention):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__(config, layer_idx)
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

        query_states = self.q_norm(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, MiniMaxM2Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM2TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
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
