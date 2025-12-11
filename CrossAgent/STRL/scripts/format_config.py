from transformers import AutoConfig, Qwen2VLConfig, Qwen2Config
import json

path = "/share/hkc/checkpoints/mc-coa-craft-qwen2vl-7b-250503_250701_rl/config.json"
with open(path, "r") as f:
    raw_config = json.load(f)

# fix: 构建 text_config 和 vision_config 为 transformers 支持的类
text_config = Qwen2Config(**raw_config["text_config"])
vision_config = AutoConfig.from_dict(raw_config["vision_config"])

# 构建主 config
main_config = Qwen2VLConfig(
    **{k: v for k, v in raw_config.items() if k not in ["text_config", "vision_config"]},
    text_config=text_config,
    vision_config=vision_config,
)

# 保存为新的 HuggingFace 格式 config
main_config.save_pretrained("/share/hkc/checkpoints/mc-coa-craft-qwen2vl-7b-250503_250701_rl")

print("✅ Config 修复完成")
