import argparse
import json
import os
import pandas as pd
import random
from openagents.agents.utils.system_prompt import mix_action_system_prompt
from copy import deepcopy
from openagents.agents.utils.action_mapping import TextActionTokenizer
from openagents.agents.utils.vlm_client import VLMClient
from rl.inference.replace_dataset import load_json_or_jsonl, get_image_message
import json
from vllm import LLM, SamplingParams
from typing import Union
from pathlib import Path
from PIL import Image
import numpy as np
from openai import OpenAI
from tqdm import tqdm

random.seed(42)
tokenizer = TextActionTokenizer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for COA-Craft.")
    parser.add_argument("--model_id", type=str, default="mc-openha-state2-qwen2-vl-7b-250830")
    parser.add_argument("--model_path", type=str, default="/share_data/limuyao/checkpoints/train/mc-openha-state2-qwen2-vl-7b-250830-A800-e1-b4-a1/checkpoints/global_step_300/hf_ckpt/") 
    parser.add_argument("--train_path", type=str, default="/public/hgx14/datasets/minecraft-trajectory.cache/mc-mix-coa_coldstart-0829.jsonl")
    parser.add_argument("--model_ips", type=str, default="localhost")
    parser.add_argument("--model_ports", type=str, default="9013,9014,9015,9016,9017,9018,9019,9020")
    parser.add_argument("--acceptable_types", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default= -1)
    parser.add_argument("--top_p", type=float, default=0.99)
    system_prompt = ""
    args = parser.parse_args()
    dataset = load_json_or_jsonl(args.train_path)
    random.shuffle(dataset)
    for idx, example in tqdm(enumerate(dataset)):
        if len(example["image"]) == 0:
            continue
        import pdb; pdb.set_trace()

        messages = []
        for i in range(len(example["conversations"])):
            for content in example["conversations"][i]["content"]:
                if content["type"] == "text":
                    message["content"].append(content)
                elif content["type"] == "image":
                    message["content"].append(get_image_message(example["image"][0]["image_path"]))
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")



        ip = args.model_ips
        port = args.model_ports.split(",")[idx % len(args.model_ports.split(","))]
        model_url = f"http://{ip}:{port}/v1"
        client = OpenAI(base_url=model_url, api_key="EMPTY")  # vLLM 默认不校验 key，填个 dummy
        completion = client.chat.completions.create(
            model=args.model_id,
            messages=[message],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body = {
                "top_k": args.top_k,
                "skip_special_tokens": False
            }
        )

        result = completion.choices[0].message.content
        ga = tokenizer.decode(ground_truth)[0]
        sa = tokenizer.decode(result)[0]


