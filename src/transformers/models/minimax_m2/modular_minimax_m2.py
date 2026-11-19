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

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs
from ..glm.modeling_glm import GlmRotaryEmbedding
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb  # noqa: F401
from ..llama.modeling_llama import LlamaRMSNorm
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
from ..olmo2.modeling_olmo2 import Olmo2Attention


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
            The epsilon used by the rms normalization layers.
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
            Whether to return router logits and compute Mixtral's generic auxiliary load-balancing loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient of Mixtral's generic auxiliary load-balancing loss in the total loss.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Parameters for rotary position embeddings. By default, `rope_theta` is 5000000.0, `rope_type` is
            `"default"`, and `partial_rotary_factor` is 0.5. Legacy `rope_theta`, `rope_scaling`, and `rotary_dim`
            arguments are also accepted.
        kwargs (`dict`, *optional*):
            Additional configuration arguments.

    ```python
    >>> from transformers import MiniMaxM2Config, MiniMaxM2Model

    >>> configuration = MiniMaxM2Config()
    >>> model = MiniMaxM2Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "minimax_m2"
    default_theta = 5_000_000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",
        "layers.*.self_attn.k_proj": "colwise_rep",
        "layers.*.self_attn.v_proj": "colwise_rep",
        "layers.*.self_attn.o_proj": "rowwise_rep",
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "local_colwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
    }

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
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        **kwargs,
    ):
        # These remote-code options describe features that are fixed or unused in this architecture.
        for key in (
            "attn_type_list",
            "num_mtp_modules",
            "mtp_transformer_layers",
            "qk_norm_type",
            "scoring_func",
            "shared_intermediate_size",
            "sliding_window",
            "router_jitter_noise",
            "use_mtp",
            "use_qk_norm",
            "use_routing_bias",
        ):
            kwargs.pop(key, None)

        rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_scaling is not None:
            rope_parameters = rope_scaling
        rope_parameters = dict(rope_parameters) if rope_parameters is not None else {}

        head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        rotary_dim = kwargs.pop("rotary_dim", None)
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", None)
        if rotary_dim is not None:
            if rotary_dim % 2 != 0:
                raise ValueError(f"`rotary_dim` must be an even number, but got {rotary_dim}.")
            if rotary_dim > head_dim:
                raise ValueError(
                    f"`rotary_dim` must be smaller than or equal to `head_dim`, but got {rotary_dim} > {head_dim}."
                )
            partial_rotary_factor = rotary_dim / head_dim
        if partial_rotary_factor is not None:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        else:
            rope_parameters.setdefault("partial_rotary_factor", 0.5)

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
            rope_parameters=rope_parameters,
            attention_dropout=attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            num_local_experts=num_local_experts,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            **kwargs,
        )
        self.attention_bias = False
        del self.sliding_window
        del self.router_jitter_noise


class MiniMaxM2RMSNorm(LlamaRMSNorm):
    pass


class MiniMaxM2RotaryEmbedding(GlmRotaryEmbedding):
    pass


class MiniMaxM2Experts(MixtralExperts):
    pass


class MiniMaxM2TopKRouter(MixtralTopKRouter):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return router_logits, top_k_weights.to(hidden_states.dtype), top_k_index


class MiniMaxM2SparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = 0.0
        self.gate = MiniMaxM2TopKRouter(config)
        self.experts = MiniMaxM2Experts(config)


class MiniMaxM2Attention(Olmo2Attention):
    pass


class MiniMaxM2DecoderLayer(MixtralDecoderLayer):
    pass


class MiniMaxM2PreTrainedModel(MixtralPreTrainedModel):
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MiniMaxM2TopKRouter):
            init.zeros_(module.e_score_correction_bias)


class MiniMaxM2Model(MixtralModel):
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MiniMaxM2ForCausalLM(MixtralForCausalLM):
    pass


__all__ = ["MiniMaxM2Config", "MiniMaxM2ForCausalLM", "MiniMaxM2Model", "MiniMaxM2PreTrainedModel"]
