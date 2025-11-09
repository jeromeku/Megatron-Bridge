# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for flop_utils."""

from types import SimpleNamespace

import pytest
import torch.nn.functional as F

from megatron.bridge.training.utils.flop_utils import num_floating_point_operations


def _build_standard_model_cfg():
    return SimpleNamespace(
        num_layers=2,
        kv_channels=64,
        num_attention_heads=8,
        hidden_size=512,
        num_query_groups=None,
        num_moe_experts=None,
        moe_layer_freq=1,
        moe_router_topk=1,
        moe_shared_expert_intermediate_size=None,
        moe_ffn_hidden_size=None,
        gated_linear_unit=False,
        activation_func=F.silu,
        multi_latent_attention=False,
        q_lora_rank=None,
        qk_head_dim=64,
        qk_pos_emb_head_dim=0,
        kv_lora_rank=0,
        v_head_dim=64,
        seq_length=16,
        vocab_size=1024,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        mtp_num_layers=None,
        ffn_hidden_size=2048,
        is_hybrid_model=False,
        hybrid_override_pattern=None,
        hybrid_attention_ratio=0.0,
        hybrid_mlp_ratio=0.0,
    )


def _build_hybrid_model_cfg():
    return SimpleNamespace(
        num_layers=6,
        seq_length=64,
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=4,
        group_query_attention=True,
        kv_channels=32,
        kv_lora_rank=0,
        v_head_dim=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=0,
        ffn_hidden_size=1024,
        gated_linear_unit=False,
        hybrid_override_pattern=["*", "M", "-", "*", "-", "M"],
        is_hybrid_model=True,
        mamba_state_dim=64,
        mamba_head_dim=32,
        mamba_num_groups=2,
        mamba_num_heads=16,
        kv_channels_override=None,
        num_moe_experts=None,
        moe_layer_freq=1,
        moe_router_topk=1,
        moe_shared_expert_intermediate_size=None,
        moe_ffn_hidden_size=None,
        activation_func=F.silu,
        multi_latent_attention=False,
        num_query_groups_override=None,
        kv_channels_attr=None,
        vocab_size=1024,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )


class TestNumFloatingPointOperations:
    """Tests for num_floating_point_operations helper."""

    def test_raises_without_cfg_or_model(self):
        """Require either cfg or model_config argument."""
        with pytest.raises(ValueError):
            num_floating_point_operations()

    @pytest.mark.parametrize(
        "builder,batch_size",
        [
            (_build_standard_model_cfg, 2),
            (_build_hybrid_model_cfg, 1),
        ],
    )
    def test_accepts_model_config_directly(self, builder, batch_size):
        """Providing a model config directly should match cfg-based behavior."""
        model_cfg = builder()
        cfg = SimpleNamespace(model=model_cfg)

        flops_from_cfg = num_floating_point_operations(cfg=cfg, batch_size=batch_size)
        flops_from_model = num_floating_point_operations(model_config=model_cfg, batch_size=batch_size)

        assert flops_from_model == flops_from_cfg
