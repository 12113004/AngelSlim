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
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

# Add Wall-OSS/lerobot to Python path if needed
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    # Try adding common paths
    LEROBOT_PATHS = [
        "/home/xieqijia/Project/WALL-OSS/lerobot/src",
    ]
    for path in LEROBOT_PATHS:
        if path not in sys.path:
            sys.path.insert(0, path)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .base_dataset import BaseDataset


class WallOSSDataset(BaseDataset):
    """Dataset for Wall-OSS VLA model calibration.

    This dataset loads LeRobot-format datasets for Wall-OSS quantization calibration.
    It handles vision-language-action data with MoE routing support.

    Expected data format:
    - LeRobot dataset with video/image observations
    - Action chunks for action prediction
    - Task instructions as text prompts
    """

    def __init__(
        self,
        processor=None,
        device: str = "cpu",
        max_length: int = 2048,
        num_samples: int = 128,
        data_source: Union[str, Dict] = None,
        model_name: str = "WallOSSVL",
        quantization_config: str = None,
        # Wall-OSS specific parameters
        pred_horizon: int = 16,
        action_dim: int = 7,
        use_fast_mode: bool = False,
    ):
        """Initialize WallOSSDataset.

        Args:
            processor: Model processor for text/image tokenization
            device: Device to load data on
            max_length: Maximum sequence length
            num_samples: Number of calibration samples to use
            data_source: Path to LeRobot dataset or dict with repo_id
            model_name: Model name (default: WallOSSVL)
            quantization_config: Quantization configuration
            pred_horizon: Prediction horizon for action chunks
            action_dim: Dimensionality of actions
            use_fast_mode: Use FAST tokenizer mode
        """
        super().__init__(processor, device, max_length)
        self.num_samples = num_samples
        self.model_name = model_name
        self.quant_algo = quantization_config.name if quantization_config else None
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.use_fast_mode = use_fast_mode

        # Load LeRobot dataset
        self._load_lerobot_dataset(data_source)

    def _load_lerobot_dataset(self, data_source: Union[str, Dict]):
        """Load LeRobot format dataset."""
        if isinstance(data_source, dict):
            repo_id = data_source.get("repo_id")
            root = data_source.get("root", None)
            episodes = data_source.get("episodes", None)
        else:
            repo_id = data_source
            root = None
            episodes = None

        # If data_source is a local path, use it directly
        if isinstance(data_source, str) and os.path.exists(data_source):
            root = data_source
            repo_id = os.path.basename(data_source)

        print(f"[WallOSSDataset] Loading LeRobot dataset: {repo_id}")

        try:
            self.lerobot_dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=episodes,
                delta_timestamps={
                    # Query current frame and next pred_horizon frames for actions
                    "action": [i / 10.0 for i in range(self.pred_horizon)],
                } if not self.use_fast_mode else None,
            )
            print(f"[WallOSSDataset] Loaded {len(self.lerobot_dataset)} frames")
        except Exception as e:
            print(f"[WallOSSDataset] Warning: Could not load LeRobot dataset: {e}")
            print("[WallOSSDataset] Creating empty dataset for testing")
            self.lerobot_dataset = None
            # Continue to create dummy data

        # Prepare calibration data
        self._prepare_calibration_data()

    def _prepare_calibration_data(self):
        """Process LeRobot data into calibration format."""
        print(f"[WallOSSDataset] Preparing {self.num_samples} calibration samples")

        if self.lerobot_dataset is None:
            # Create dummy data for testing
            self._create_dummy_data()
            return

        num_samples = min(self.num_samples, len(self.lerobot_dataset))

        for idx in tqdm(range(num_samples), desc="Loading calibration data"):
            try:
                sample = self.lerobot_dataset[idx]
                processed = self._process_sample(sample)
                if processed is not None:
                    self.data.append(processed)
            except Exception as e:
                print(f"[WallOSSDataset] Warning: Failed to process sample {idx}: {e}")
                continue

        print(f"[WallOSSDataset] Prepared {len(self.data)} valid samples")

    def _create_dummy_data(self):
        """Create dummy data for testing when no dataset is available."""
        print("[WallOSSDataset] Creating dummy calibration data")

        for i in range(min(self.num_samples, 10)):
            # Create minimal dummy data that matches expected format
            dummy_data = {
                "input_ids": torch.randint(0, 32000, (1, 128)),
                "attention_mask": torch.ones(1, 128, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 224, 224),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
                # MoE routing info
                "moe_token_types": torch.zeros(1, 128, dtype=torch.long),
                # Action-related (will be handled by model)
                "action": torch.randn(1, self.pred_horizon, self.action_dim),
            }
            self.data.append(dummy_data)

    def _process_sample(self, sample: Dict) -> Optional[Dict]:
        """Process a single LeRobot sample into model input format.

        Args:
            sample: LeRobot dataset sample

        Returns:
            Processed sample dict or None if invalid
        """
        try:
            # Extract images
            images = self._extract_images(sample)

            # Extract task instruction
            task = sample.get("task", "")
            if not task:
                # Default task if not provided
                task = "Perform the robot manipulation task."

            # Extract actions
            action = sample.get("action", None)
            if action is None:
                return None

            # Prepare messages for processor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0] if images else None},
                        {"type": "text", "text": task},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'll perform the task."}],
                },
            ]

            # Process with model processor if available
            if self.processor is not None:
                inputs = self._process_with_processor(messages)
            else:
                # Create basic inputs without processor
                inputs = self._create_basic_inputs(images, task)

            # Add MoE token types for routing
            inputs["moe_token_types"] = self._create_moe_token_types(inputs)

            # Add actions for calibration
            if isinstance(action, torch.Tensor):
                inputs["action"] = action.unsqueeze(0) if action.dim() == 1 else action

            return inputs

        except Exception as e:
            print(f"[WallOSSDataset] Error processing sample: {e}")
            return None

    def _extract_images(self, sample: Dict) -> List[Image.Image]:
        """Extract images from LeRobot sample."""
        images = []

        # Common image keys in LeRobot datasets
        image_keys = [
            "observation.images.cam_high",
            "observation.images.cam_low",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
            "observation.image",
        ]

        for key in image_keys:
            if key in sample:
                img_data = sample[key]
                if isinstance(img_data, torch.Tensor):
                    # Convert tensor to PIL Image
                    img_tensor = img_data.cpu()
                    if img_tensor.dim() == 3:
                        img_array = img_tensor.permute(1, 2, 0).numpy()
                        # Normalize if needed
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype('uint8')
                        images.append(Image.fromarray(img_array))
                elif isinstance(img_data, Image.Image):
                    images.append(img_data)

        return images

    def _process_with_processor(self, messages: List[Dict]) -> Dict:
        """Process messages with model processor."""
        try:
            # Use Qwen2.5-VL style processing
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # max_length padding for int4 awq
            if self.quant_algo and "int4_" in self.quant_algo:
                padding = "max_length"
            else:
                padding = True

            inputs = self.processor(
                text=[text],
                images=None,  # Images already in messages
                padding=padding,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )

            # Convert to dict of tensors
            return {k: v for k, v in inputs.items()}

        except Exception as e:
            print(f"[WallOSSDataset] Processor failed: {e}, using basic inputs")
            return self._create_basic_inputs([], "")

    def _create_basic_inputs(self, images: List, text: str) -> Dict:
        """Create basic inputs without processor."""
        return {
            "input_ids": torch.randint(0, 32000, (1, self.max_length)),
            "attention_mask": torch.ones(1, self.max_length, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 224, 224) if images else torch.randn(1, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def _create_moe_token_types(self, inputs: Dict) -> torch.Tensor:
        """Create MoE token type IDs for routing.

        Wall-OSS uses MoE routing with different token types:
        - 0: text tokens
        - 1: image tokens
        - 2: action tokens
        """
        seq_len = inputs.get("input_ids", torch.zeros(1, 128)).shape[1]

        # For now, mark all as text tokens (0)
        # In practice, this would be set based on actual token types
        moe_token_types = torch.zeros(1, seq_len, dtype=torch.long)

        # Mark image positions if image_grid_thw is present
        if "image_grid_thw" in inputs:
            # Image tokens typically come after the initial text tokens
            # This is model-specific and may need adjustment
            pass

        return moe_token_types

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range")
        return self.data[idx]

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for batching Wall-OSS samples."""
        collated = {}

        for key in batch[0].keys():
            values = [item[key] for item in batch]

            if isinstance(values[0], torch.Tensor):
                # Handle different tensor dimensions
                if all(t.shape == values[0].shape for t in values):
                    # All same shape - stack them
                    if values[0].dim() == 0:
                        collated[key] = torch.stack(values)
                    else:
                        collated[key] = torch.cat(values, dim=0)
                else:
                    # Variable shapes - keep as list or pad
                    collated[key] = values
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            elif isinstance(values[0], str):
                collated[key] = values
            else:
                collated[key] = values

        return collated
