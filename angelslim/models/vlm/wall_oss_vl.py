# Copyright 2025 Tencent Inc. All Rights Reserved.
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

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

# Add Wall-OSS to Python path
WALL_OSS_PATH = "/home/xieqijia/Project/WALL-OSS/wall-x_eager_attention"
if WALL_OSS_PATH not in sys.path:
    sys.path.insert(0, WALL_OSS_PATH)

# Import transformers first for monkey-patch
from transformers import PretrainedConfig

# Monkey-patch Qwen2_5_VLConfig.from_pretrained to fix text_config
_original_from_pretrained = None

def _patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Patched version that fixes text_config from dict to PretrainedConfig."""
    config = _original_from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Fix: Convert text_config from dict to PretrainedConfig if needed
    if hasattr(config, 'text_config') and isinstance(config.text_config, dict):
        text_config_obj = PretrainedConfig.from_dict(config.text_config)
        config.text_config = text_config_obj
        print(f"[WallOSSVL] Fixed text_config from dict to PretrainedConfig")

    return config

# Import Wall-OSS model
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLMoEForAction,
    SparseMoeBlock,
)
from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLConfig

# Apply monkey-patch
_original_from_pretrained = Qwen2_5_VLConfig.from_pretrained
Qwen2_5_VLConfig.from_pretrained = classmethod(_patched_from_pretrained)

from transformers import AutoProcessor

from ...utils import find_layers, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class WallOSSVL(BaseLLMModel):
    """
    Wall-OSS Vision-Language-Action model support for AngelSlim.

    Wall-OSS is based on Qwen2.5-VL with Mixture of Experts (MoE) architecture,
    designed for vision-language-action tasks with support for both diffusion
    and fast action prediction modes.
    """

    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.modal_type = "VLM"
        # Path to transformer layers in Wall-OSS
        self.block_name = "model.layers"
        self.vit_block_name = "visual.blocks"

        # Modules to exclude from quantization (visual and embedding components)
        self.pre_transformer_module_names = [
            "visual",
            "embed_tokens",
            "norm",
            "rotary_emb",
            "lm_head",
        ]

        # Observer layer classes - include MoE blocks for Wall-OSS
        self.observer_layer_classes = [nn.Linear, SparseMoeBlock]

        # Layer names to observe for quantization
        self.observed_names = [
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "out_proj",
        ]

    def _patch_attention_for_awq(self):
        """
        Patch attention layers to handle None position_embeddings during AWQ calibration.

        AWQ's Catcher hook captures layer_kwargs which may have position_embeddings=None.
        This patch ensures that if position_embeddings is None, we dynamically generate it
        using the model's rotary_emb.
        """
        print_info("[Slim] Patching attention layers for AWQ compatibility")

        # Get rotary_emb from the model
        rotary_emb = self.model.model.rotary_emb

        def create_patched_forward(original_forward):
            """Create a patched forward that handles None position_embeddings."""
            def patched_forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Any] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs
            ):
                # If position_embeddings is None, generate it dynamically
                if position_embeddings is None:
                    # Get position_ids if not provided
                    if position_ids is None:
                        # Create default position_ids: (3, batch, seq_len)
                        batch_size, seq_len, _ = hidden_states.shape
                        device = hidden_states.device
                        position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)
                    else:
                        # Ensure position_ids is on the same device as hidden_states
                        position_ids = position_ids.to(hidden_states.device)

                    # Generate position_embeddings using rotary_emb
                    # Ensure rotary_emb is on the same device as hidden_states
                    with torch.no_grad():
                        if rotary_emb.inv_freq.device != hidden_states.device:
                            rotary_emb.to(hidden_states.device)
                        position_embeddings = rotary_emb(hidden_states, position_ids)

                # Call original forward with valid position_embeddings
                return original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs
                )
            return patched_forward

        # Patch all attention layers in the model
        patched_count = 0
        for layer in self.model.model.layers:
            if hasattr(layer, 'self_attn'):
                original_forward = layer.self_attn.forward
                layer.self_attn.forward = create_patched_forward(original_forward)
                patched_count += 1

        print_info(f"[Slim] Patched {patched_count} attention layers")

    def _patch_moe_for_awq(self):
        """
        Patch MoE layers to handle None token_types/start_indices/end_indices during AWQ calibration.

        AWQ calls layers directly without going through the model's forward, so MoE routing
        parameters are not computed. This patch provides default routing when these are None.

        For Wall-OSS with 2 experts (standard + shared), we route tokens to both experts
        to ensure AWQ can capture inputs for all expert layers.
        """
        print_info("[Slim] Patching MoE layers for AWQ compatibility")

        def create_patched_moe_forward(moe_module, original_forward):
            """Create a patched MoE forward that handles None routing parameters."""
            num_experts = len(moe_module.experts) if hasattr(moe_module, 'experts') else 8

            def patched_moe_forward(hidden_states, experts_indices=None, start_indices=None, end_indices=None):
                # If routing parameters are None, create default routing
                if experts_indices is None:
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    total_tokens = batch_size * seq_len

                    # Route tokens to all experts for calibration
                    # Expert 0 gets first half, Expert 1 (shared) gets second half
                    # This ensures AWQ can capture inputs for all expert layers
                    experts_indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
                    mid_point = total_tokens // 2

                    # Flatten and assign: first half to expert 0, second half to expert 1
                    flat_indices = experts_indices.view(-1)
                    flat_indices[mid_point:] = 1  # Assign second half to expert 1
                    experts_indices = flat_indices.view(batch_size, seq_len)

                    # Create start/end indices for all experts
                    start_indices = torch.zeros(num_experts, dtype=torch.long, device=hidden_states.device)
                    end_indices = torch.zeros(num_experts, dtype=torch.long, device=hidden_states.device)

                    # Expert 0: [0, mid_point)
                    start_indices[0] = 0
                    end_indices[0] = mid_point

                    # Expert 1: [mid_point, total_tokens)
                    if num_experts > 1:
                        start_indices[1] = mid_point
                        end_indices[1] = total_tokens

                return original_forward(hidden_states, experts_indices, start_indices, end_indices)
            return patched_moe_forward

        # Patch all MoE layers in the model
        patched_count = 0
        for layer in self.model.model.layers:
            if hasattr(layer, 'moe') and layer.moe is not None:
                original_forward = layer.moe.forward
                layer.moe.forward = create_patched_moe_forward(layer.moe, original_forward)
                patched_count += 1

        print_info(f"[Slim] Patched {patched_count} MoE layers")

    def from_pretrained(
        self,
        model_path,
        train_config=None,
        config_path=None,
        action_tokenizer_path=None,
        use_head_dim_padding=False,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        using_multi_nodes=False,
        **kwargs,  # Accept additional arguments from engine
    ):
        """
        Load Wall-OSS model from pretrained path.

        Args:
            model_path: Path to the pretrained model
            train_config: Training configuration dict (YAML loaded)
            config_path: Optional path to config file
            action_tokenizer_path: Path to action tokenizer (e.g., FAST tokenizer)
            use_head_dim_padding: Whether to use head_dim padding
            torch_dtype: Data type for model weights
            device_map: Device mapping for model loading
            trust_remote_code: Whether to trust remote code
            low_cpu_mem_usage: Whether to use low CPU memory
            use_cache: Whether to use KV cache
            using_multi_nodes: Whether using multi-node setup
        """
        print_info(f"[Slim] Loading Wall-OSS model from {model_path}")

        # Load train_config if not provided
        if train_config is None:
            import yaml
            default_config_path = "/home/xieqijia/Project/WALL-OSS/wall-x/workspace/libero/config_qact.yml"
            if os.path.exists(default_config_path):
                with open(default_config_path, 'r') as f:
                    train_config = yaml.safe_load(f)
                print_info(f"[Slim] Loaded train_config from {default_config_path}")
            else:
                raise ValueError("train_config must be provided or config file must exist")

        # Set action_tokenizer_path if not provided
        if action_tokenizer_path is None:
            action_tokenizer_path = "/home/xieqijia/Models/fast"

        # Load model using Wall-OSS specific loader
        # Note: Monkey-patch above fixes text_config from dict to PretrainedConfig
        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            pretrained_model_path=model_path,
            train_config=train_config,
            config_path=config_path,
            action_tokenizer_path=action_tokenizer_path,
            use_head_dim_padding=use_head_dim_padding,
        )

        # Set model to evaluation mode
        self.model.eval()

        # Store config reference
        self.config = self.model.config

        # Load processor for VLM processing
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
            print_info(f"[Slim] Processor loaded successfully")
        except Exception as e:
            print_info(f"[Slim] Warning: Could not load processor: {e}")
            self.processor = None

        # Patch attention layers for AWQ compatibility
        self._patch_attention_for_awq()
        self._patch_moe_for_awq()

        print_info(f"[Slim] Wall-OSS model loaded successfully")
        print_info(f"[Slim] Model has {self.config.num_hidden_layers} layers with MoE")

    def get_observer_layers(self):
        """
        Get layers to observe for quantization.

        For Wall-OSS MoE model, we observe linear layers and MoE blocks
        in the transformer layers while excluding visual and embedding layers.
        """
        names = self.quant_config.quant_algo_info.get("ignore_layers", [])

        # Find all observer layers in the model
        observer_layers_dict = find_layers(
            self.model,
            layers=self.observer_layer_classes
        )

        # Filter to only include layers in transformer blocks
        observer_layers_dict = {
            k: v
            for k, v in observer_layers_dict.items()
            if k.startswith(self.block_name) and not any(name in k for name in names)
        }

        # Filter by observed names if specified
        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in list(observer_layers_dict.keys()):
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name, None)

        print_info(f"[Slim] Found {len(observer_layers_dict)} observer layers")
        return observer_layers_dict

    def get_quant_module(self):
        """
        Returns the module that will be quantized.

        For Wall-OSS, this is the main transformer layers.
        """
        return self.model.model.layers

    def model_forward(self, dataloader, **kwargs):
        """
        Run forward pass for calibration.

        For Wall-OSS, we use mode="predict" with predict_mode="diffusion"
        to match the inference pattern.
        """
        self.model.eval()
        calibrated_cnt = 0

        device = next(self.model.parameters()).device
        print_info(f"[Slim] Calibration device: {device}")

        if dataloader is not None:
            with torch.no_grad():
                for batch in dataloader:
                    try:
                        # Move batch to device
                        inputs = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }

                        # Use predict mode for calibration (matching inference)
                        outputs = self.model(
                            mode="predict",
                            predict_mode="diffusion",
                            **inputs
                        )

                        calibrated_cnt += 1
                        if calibrated_cnt % 10 == 0:
                            print_info(f"[Slim] Calibrated {calibrated_cnt} batches")

                    except Exception as e:
                        print_info(f"[Slim] Warning: calibration batch failed: {e}")
                        calibrated_cnt += 1
                        continue

        print_info(f"[Slim] Calibration complete: {calibrated_cnt} batches")

    def get_save_func(self):
        """
        Get the save function for quantized model.

        For now, we use the standard VLM save. This may need customization
        for Wall-OSS specific checkpoint format.
        """
        from ...compressor.quant.core import PTQVLMSaveVllmHF

        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQVLMSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
