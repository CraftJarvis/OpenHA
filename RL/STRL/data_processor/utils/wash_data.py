input_path = "OpenHA/rl/datasets/sft/msrl_coldstart/20250909/mc-mix-coa-qwen2-vl-7b-250906_120-withsystem-onlyraw.jsonl"
output_path = "OpenHA/rl/datasets/sft/msrl_coldstart/20250909/mc-mix-coa-qwen2-vl-7b-250906_120-withsystem-onlyraw-washed.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        # 删除所有 <|im_start|>
        cleaned = line.replace("<|im_start|>", "").replace("<|endoftext|>", "").replace("<|im_end|>", "")
        fout.write(cleaned)
