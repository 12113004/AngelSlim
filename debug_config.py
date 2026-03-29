"""Debug script for WallOSSVL model loading issue."""

import sys
import torch
import yaml

# Add Wall-OSS to path
WALL_OSS_PATH = "/home/xieqijia/Project/WALL-OSS/wall-x_eager_attention"
if WALL_OSS_PATH not in sys.path:
    sys.path.insert(0, WALL_OSS_PATH)

# Add lerobot to path
LERO_PATH = "/home/xieqijia/Project/WALL-OSS/lerobot/src"
if LERO_PATH not in sys.path:
    sys.path.insert(0, LERO_PATH)

from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

# Load the config
config_path = "/home/xieqijia/Project/WALL-OSS/wall-x/workspace/libero/workspace/finetuned_new/config.json"
config = Qwen2_5_VLConfig.from_pretrained(config_path)

print("Config type:", type(config))
print("\nConfig attributes:")
for attr in dir(config):
    if not attr.startswith("_"):
        try:
            val = getattr(config, attr)
            if not callable(val):
                print(f"  {attr}: {type(val).__name__}")
        except:
            pass

print("\n" + "="*60)
print("Checking text_config...")
if hasattr(config, 'text_config'):
    text_config = config.text_config
    print(f"text_config type: {type(text_config)}")
    if isinstance(text_config, dict):
        print("text_config is a dict - THIS IS THE PROBLEM!")
        print("Keys:", list(text_config.keys()))
    else:
        print("text_config is an object - should work")
        print("Has to_dict:", hasattr(text_config, 'to_dict'))
        if hasattr(text_config, 'to_dict'):
            print("to_dict() works:", text_config.to_dict())

# Try to find a workaround
print("\n" + "="*60)
print("Trying to fix text_config...")

if hasattr(config, 'text_config') and isinstance(config.text_config, dict):
    # Option 1: Create a PretrainedConfig from dict
    try:
        from transformers import PretrainedConfig
        text_config_obj = PretrainedConfig.from_dict(config.text_config)
        print(f"Created text_config object: {type(text_config_obj)}")
        print("Has to_dict:", hasattr(text_config_obj, 'to_dict'))

        # Replace the text_config with the object
        config.text_config = text_config_obj
        print("Replaced text_config with object")

        # Check get_text_config
        if hasattr(config, 'get_text_config'):
            decoder_config = config.get_text_config(decoder=True)
            print(f"get_text_config returned: {type(decoder_config)}")

    except Exception as e:
        print(f"Failed to create PretrainedConfig: {e}")

    # Option 2: Check if there's a nested text_config in the outer config
    print("\nChecking for get_text_config method...")
    if hasattr(config, 'get_text_config'):
        print("get_text_config exists")
        try:
            result = config.get_text_config(decoder=True)
            print(f"Result type: {type(result)}")
        except Exception as e:
            print(f"get_text_config failed: {e}")

print("\n" + "="*60)
print("Full config structure:")
import json
print(json.dumps(config.to_dict(), indent=2, default=str)[:2000])
