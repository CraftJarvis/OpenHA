import random
from pathlib import Path

# 3个源文件路径
files = [
    "OpenHA/rl/datasets/sft/msrl_coldstart/20250923/mc-msrl_coldstart-grounding_250923-train-withsystem.jsonl",
    "OpenHA/rl/datasets/sft/msrl_coldstart/20250923/mc-msrl_coldstart-motion_250923-train-withsystem.jsonl",
    "OpenHA/rl/datasets/sft/msrl_coldstart/20250923/mc-msrl_coldstart-text_250923-train-withsystem.jsonl"
]

# 输出文件
output_path = "OpenHA/rl/datasets/sft/msrl_coldstart/20250923/mc-msrl_coldstart-sampled_10000-each.jsonl"

# 每个文件抽取的行数
num_samples = 10000

all_samples = []

for fpath in files:
    with open(fpath, "r") as f:
        lines = f.readlines()
    print(f"{fpath} 共 {len(lines)} 行")

    # 如果行数少于num_samples，则全取
    if len(lines) <= num_samples:
        sampled = lines
    else:
        sampled = random.sample(lines, num_samples)

    all_samples.extend(sampled)

# 将所有样本写入新文件
with open(output_path, "w") as out:
    out.writelines(all_samples)

print(f"✅ 已完成抽样，每个文件取 {num_samples} 行，共 {len(all_samples)} 行")
print(f"输出文件：{output_path}")
