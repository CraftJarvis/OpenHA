import re
from tqdm import tqdm
from openagents.agents.utils.system_prompt import coa_system_prompt
from openagents.agents.utils.system_prompt import mix_action_system_prompt
import math
import torch
from minestudio.simulator.entry import MinecraftSim
import numpy as np
from copy import deepcopy
import random
from typing import Union
from openagents.agents.utils.action_mapping import TextActionTokenizer
from pathlib import Path
from PIL import Image
import base64
from openai import OpenAI
from transformers import AutoTokenizer
from copy import deepcopy
import ray
from rl.datasets.utils.data_preprocess import get_process_func, load_json_or_jsonl
import ray
from ray.util import ActorPool
from transformers import AutoTokenizer

@ray.remote
class InferenceWorker:
    def __init__(self, port, args, system_prompt, tokenizer_path):
        from openai import OpenAI
        from openagents.agents.utils.action_mapping import TextActionTokenizer
        self.client = OpenAI(base_url=f"http://{args.model_ips}:{port}/v1", api_key="EMPTY")
        self.autotokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tokenizer = TextActionTokenizer()
        self.args = args
        self.system_prompt = system_prompt
        self.process_one_func = get_process_func(args.process_type)
        
    def run(self, example, idx):
        return self.process_one_func(
            example, idx,
            self.args, self.system_prompt,
            self.autotokenizer, self.tokenizer, self.client
        )

def reject_sampling(dataset, output_path, args, system_prompt):
    ray.init(ignore_reinit_error=True)

    ports = args.model_ports.split(",")
    # 启动8个actor，每个actor连一个port
    # 16 个 Actor，每个选择一个端口（循环分配）这行吗

    #test
    process_one = get_process_func(args.process_type)
    for i in range(1):
        ex, suffix = process_one(
            dataset[i], i,
            args, system_prompt,
            autotokenizer, TextActionTokenizer(), None
        )
        print("Test example:", ex)
        print("Test suffix:", suffix)
    # test finish

    

    workers = [
        InferenceWorker.remote(ports[i % len(ports)], args, system_prompt, args.model_path)
        for i in range(1)
    ]

    pool = ActorPool(workers)

    suc_nums = {"Motion:": 0, "Action:": 0, "Grounding:": 0}
    total = len(dataset)

    if args.cover and Path(output_path).exists():
        print("覆盖已存在的文件:", output_path)
        Path(output_path).unlink()

    print("keep_failed:", args.keep_failed)
    print("输出文件:", output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        for i, (ex, extra_info) in tqdm(enumerate(pool.map_unordered(
                lambda a, data: a.run.remote(data[1], data[0]),
                list(enumerate(dataset))
            )), total = total, desc="Reject Sampling"
        ):
            if ex is not None:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                suffix = extra_info.get("suffix", None)
                if suffix:
                    suc_nums[suffix] += 1
            else:
                if args.keep_failed:
                    f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")
            
            if (i + 1) % 100 == 0:
                print("final success nums:", suc_nums)
                suc_num = sum(suc_nums.values())
                print("final success rate:", suc_num / (i + 1))



if __name__ == "__main__":
    import argparse
    import json
    import os
    import pandas as pd
    import random
    from datetime import datetime

    # ssrl cold start 里不成功的直接扔掉。
    # msrl cold start 里不成功的保留

    parser = argparse.ArgumentParser(description="Process dataset for COA-Craft.")
    parser.add_argument("--train_path", type=str, default="/public/hgx14/datasets/minecraft-trajectory.cache/mc-mix-coa_coldstart-0906.jsonl")
    parser.add_argument("--model_id", type=str, default="mc-openha-state2-qwen2-vl-7b-250830")
    parser.add_argument("--model_path", type=str, default="/share_data/limuyao/checkpoints/train/mc-openha-state2-qwen2-vl-7b-250830-A800-e1-b4-a1/checkpoints/global_step_300/hf_ckpt/")
    parser.add_argument("--output_path", type=str, default="OpenHA/rl/datasets/sft/toy")
    parser.add_argument("--keep_failed", type=bool, default=True, help="是否保留处理失败的样本")
    parser.add_argument("--cover", type=bool, default=False, help="是否覆盖已存在的文件")
    parser.add_argument("--only_raw_history", type=bool, default=False, help="是否只保留raw action history")
    parser.add_argument("--model_ips", type=str, default="localhost")
    parser.add_argument("--model_ports", type=str, default="9013,9014,9015,9016,9017,9018,9019,9020")
    parser.add_argument("--acceptable_types", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default= -1)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--process_type", type=str, default="mix_coa", help="选择处理函数")

    args = parser.parse_args() 
    autotokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    system_prompt = mix_action_system_prompt


    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(args.output_path, f"{timestamp}/{args.model_id}.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = load_json_or_jsonl(args.train_path)
    random.shuffle(dataset)

    reject_sampling(dataset, output_path, args, system_prompt)

    
            