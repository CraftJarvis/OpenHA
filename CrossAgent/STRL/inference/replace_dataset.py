import argparse
import json
import os
import pandas as pd
import random
from openagents.agents.utils.system_prompt import mix_action_system_prompt
from copy import deepcopy
from openagents.agents.utils.action_mapping import TextActionTokenizer
from openagents.agents.utils.vlm_client import VLMClient
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

def load_json_or_jsonl(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")

def parse_history2action(example, history_length=0):
    assert 2*(history_length+1) <= len(example["conversations"]), "Conversations length mismatch."
    example_new = deepcopy(example)
    example_new["conversations"] = [example["conversations"][0]]
    example_new["conversations"].extend(example["conversations"][len(example["conversations"])-2*history_length-1:])  # 保留最后两轮对话
    example_new["image"] = example["image"][-history_length-1:]  # 保留最后一张图片

    raw_actions = []
    assert len(example["conversations"])%2 ==0, "Conversations length should be even."
    for i in range(len(example["conversations"])//2):
        raw_actions.append(tokenizer.decode(example["conversations"][2*i+1]["content"][0]["text"])[0])

    return example_new

def get_suffix(image:Union[list,str,Path,np.ndarray,Image.Image]):
    if isinstance(image,np.ndarray|Image.Image):
        image_suffix = 'jpeg'
    elif isinstance(image,str):
        image_suffix = image.split(".")[-1]
    elif isinstance(image,Path):
        image_suffix = image.suffix[1:]
    else:
        raise ValueError(f"invalid image type！")
    return image_suffix

def get_image_message(source_data:Union[str,Path,np.ndarray,Image.Image]):
    image_suffix = get_suffix(source_data)
    image = { "url": f"data:image/{image_suffix};base64,{encode_image_to_base64(source_data)}"}
    image_message = {
            "type": "image_url",
            "image_url": image,
        }
    return image_message

import base64

def encode_image_to_base64(image_path: str) -> str:
    """读取本地图片文件并转成 Base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

import re
from typing import Dict, Optional

def parse_actions(text: str) -> Dict[str, Optional[str]]:
    """
    从输入字符串中解析出 Grounding / Motion / Action 的内容。
    每种类型至多出现一次，缺失则为 None。
    """
    result = {"Grounding": None, "Motion": None, "Action": None}
    
    pattern = re.compile(r"(Grounding|Motion|Action):\s*(.*?)(?=(?:Grounding:|Motion:|Action:|$))", re.DOTALL)
    for key, val in pattern.findall(text):
        result[key] = val.strip(" ,\n")
    
    return result



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process dataset for COA-Craft.")
    parser.add_argument("--model_id", type=str, default="mc-mix-01reward-coa-qwen2-vl-7b-250825")
    parser.add_argument("--model_path", type=str, default="/share/hkc/checkpoints/ssrl/mc-mix-01reward-coa-qwen2-vl-7b-250825/global_step_20") 
    parser.add_argument("--train_path", type=str, default="/share/hkc/datasets/mc-mix-action-0816-train.jsonl")
    parser.add_argument("--output_path", type=str, default="OpenHA/rl/datasets/sft/ss/mix_after_rl.json")
    parser.add_argument("--model_ips", type=str, default="localhost")
    parser.add_argument("--model_ports", type=str, default="9013,9014,9015,9016,9017,9018,9019,9020")
    parser.add_argument("--acceptable_types", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default= -1)
    parser.add_argument("--top_p", type=float, default=0.99)
    system_prompt = mix_action_system_prompt
    args = parser.parse_args()


    dataset = load_json_or_jsonl(args.train_path)
    # with open(f"OpenHA/rl/datasets/sft/ss_coa_splits/motions.json","a") as f1:
    #     with open(f"OpenHA/rl/datasets/sft/ss_coa_splits/groundings.json","a") as f2:
    #         with open(f"OpenHA/rl/datasets/sft/ss_coa_splits/actions.json","a") as f3:
    #             for idx, example in tqdm(enumerate(dataset)):
    #                 if len(example['image'])==0: 
    #                     continue
    #                 processed_data = parse_history2action(example, history_length=0)
    #                 ground_truth = example["conversations"][-1]["content"][0]["text"]
    #                 action_dict = parse_actions(ground_truth)
    #                 if action_dict["Motion"] is not None:
    #                     f1.write(json.dumps(processed_data)+"\n")
    #                 elif action_dict["Grounding"] is not None:
    #                     f2.write(json.dumps(processed_data)+"\n")
    #                 else:
    #                     if "Motion" in ground_truth:
    #                         import pdb; pdb.set_trace()
    #                     f3.write(json.dumps(processed_data)+"\n")

    with open(f"{args.output_path}", "a") as f:
        random.shuffle(dataset)
        for idx, example in tqdm(enumerate(dataset)):
            if len(example["image"]) == 0:
                continue
            processed_data = parse_history2action(example, history_length=0)
            ground_truth = processed_data["conversations"][-1]["content"][0]["text"]
            
            #task_group, coa_group = parse_datagroup(processed_data)

            message = {"role": "user", "content": []}
            for content in processed_data["conversations"][0]["content"]:
                if content["type"] == "text":
                    content["text"] = system_prompt+content["text"]
                    message["content"].append(content)
                elif content["type"] == "image":
                    message["content"].append(get_image_message(processed_data["image"][0]["image_path"]))
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
            replace = 1
            for key in sa:
                if key == "ESC":
                    continue
                if key == "camera":
                    if not np.equal(sa[key], ga[key]).all():
                        replace = 0
                elif sa[key] != ga[key]:
                    replace = 0
                else:
                    pass
            print("replace:", replace)
            if replace == 1:
                processed_data["conversations"][-1]["content"][0]["text"] = result

            f.write(json.dumps(processed_data) + "\n")