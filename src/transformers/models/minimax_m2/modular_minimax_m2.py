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
from torch.nn import functional as F
from torch.nn import init

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ..glm.modeling_glm import GlmRotaryEmbedding
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb  # noqa: F401
from ..llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm
from ..mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralExperts,
    MixtralPreTrainedModel,
    MixtralTopKRouter,
)
from ..olmo2.modeling_olmo2 import Olmo2Attention
from ..olmoe.modeling_olmoe import OlmoeModel
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


class MiniMaxM2Config(PreTrainedConfig):
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
            `num_key_value_heads=1`, it uses multi-query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
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
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts selected for each token.
        num_local_experts (`int`, *optional*, defaults to 256):
            Number of experts in each sparse MLP layer.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Parameters for rotary position embeddings. By default, `rope_type` is `"default"` and
            `partial_rotary_factor` is 0.5, so rotary embeddings are applied to half of each attention head.
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
    keys_to_ignore_at_inference = ["past_key_values"]
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
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_experts": "num_local_experts",
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
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 8,
        num_local_experts: int | None = 256,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.rope_parameters = rope_parameters if rope_parameters is not None else {}
        self.rope_parameters.setdefault(
            "partial_rotary_factor", kwargs.pop("rotary_dim", self.head_dim // 2) / self.head_dim
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


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
        # The only difference from MixtralTopKRouter is ungrouped sigmoid routing with a correction bias.
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return router_logits, top_k_weights.to(hidden_states.dtype), top_k_index


class MiniMaxM2SparseMoeBlock(Qwen3MoeSparseMoeBlock):
    pass


class MiniMaxM2Attention(Olmo2Attention):
    pass


class MiniMaxM2DecoderLayer(MixtralDecoderLayer):
    pass


class MiniMaxM2PreTrainedModel(MixtralPreTrainedModel):
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _can_record_outputs = {
        "hidden_states": MiniMaxM2DecoderLayer,
        "attentions": MiniMaxM2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MiniMaxM2TopKRouter):
            init.zeros_(module.e_score_correction_bias)


class MiniMaxM2Model(OlmoeModel):
    pass


class MiniMaxM2ForCausalLM(LlamaForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniMaxM2ForCausalLM

        >>> model = MiniMaxM2ForCausalLM.from_pretrained("MiniMaxAI/MiniMax-M2.5")
        >>> tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2.5")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> generate_ids = model.generate(inputs.input_ids, max_new_tokens=30)
        ```"""
        return super().forward(**super_kwargs)


__all__ = ["MiniMaxM2Config", "MiniMaxM2ForCausalLM", "MiniMaxM2Model", "MiniMaxM2PreTrainedModel"]
