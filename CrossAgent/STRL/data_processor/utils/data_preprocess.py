from copy import deepcopy
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
import json
from rl.data_processor.utils.instructions import get_instruction, get_task_type, get_task_from_ins
from vllm import LLM, SamplingParams
def encode_image_to_base64(image_path: str) -> str:
    """读取本地图片文件并转成 Base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def to_text_only(messages):
    """Convert multimodal messages (text + image_url) into pure text-only messages."""
    text_messages = []
    for msg in messages:
        new_content = []
        for item in msg.get("content", []):
            if item["type"] == "text":
                new_content.append(item["text"])
            elif item["type"] in ("image", "image_url"):
                new_content.append("[Image omitted]")  # 或 "[图片省略]" / f"[Image: {short_name}]"
        text_messages.append({
            "role": msg["role"],
            "content": "\n".join(new_content)
        })
    return text_messages

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

def load_json_or_jsonl(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")

def judge_equal_action(sa, gt):
    for key in sa:
        if key == "ESC":
            continue
        elif key == "camera":
            if not np.equal(sa[key], gt[key]).all():
                return 0
        elif sa[key] != gt[key]:
            return 0
        else:
           continue
    return 1

def get_image_message(source_data:Union[str,Path,np.ndarray,Image.Image]):
    image_suffix = get_suffix(source_data)
    image = { "url": f"data:image/{image_suffix};base64,{encode_image_to_base64(source_data)}"}
    image_message = {
            "type": "image_url",
            "image_url": image,
        }
    return image_message

def check_suffix(response):
    if "Motion:" in response and "Grounding:" in response:
        return None
    elif "Motion:" in response:
        return "Motion:"
    elif "Grounding:" in response:
        return "Grounding:"
    elif "Action:" in response:
        return "Action:"
    else:
        return None


# only raw history, dont change solution
def parse_msg_mix_coa(messages_ogn, images_ogn, only_raw_history, system_prompt, maximum_history_length = 5):

    messages = deepcopy(messages_ogn)
    images = deepcopy(images_ogn)

    if "conversations" in messages:
        messages = messages["conversations"]
    
    messages = [messages[0]] + messages[1:][-2*maximum_history_length-1:]
    images = images[-maximum_history_length-1:]
    image_id = 0
    for i, conversation in enumerate(messages[:-1]):
        for j, content in enumerate(conversation["content"]):
            if "image" in content["type"]:
                if image_id >= len(images):
                    continue
                image_path = images[image_id]["image_path"]
                image_path = image_path.replace("/DATA", "/public/hgx14")
                content = get_image_message(image_path)
                image_id += 1
            if only_raw_history and content["type"] == "text"  and "Action:" in content["text"]:
                content["text"] = "Action:" + content["text"].split("Action:")[1]
            messages[i]["content"][j] = content
    if image_id != len(images):
        import pdb; pdb.set_trace()

    if system_prompt not in messages[0]["content"][0]["text"]:
        messages[0]["content"][0]["text"] = system_prompt + messages[0]["content"][0]["text"]

    else:
        raise NotImplementedError(f"Data type '{data_type}' is not supported.")

    return messages, images

def process_one_mix_coa(example, idx, args, system_prompt, autotokenizer, tokenizer, client, filter_noop=True):
    """并行处理一条数据，返回 (processed_example, suffix) 或 (None, None)"""
    try:
        extra_info = {}
        if len(example["image"]) == 0:
            return None, extra_info

        # 注意：Ray 里的函数不能直接用外部对象，需要自己加载
        from openagents.agents.utils.system_prompt import mix_action_system_prompt
        from openagents.agents.utils.action_mapping import TextActionTokenizer
        from copy import deepcopy

        instruction = get_instruction(example["conversations"])
        task = get_task_from_ins(instruction)
        extra_info["task"] = task

        messages = parse_msg_mix_coa(example["conversations"], example["image"], args.only_raw_history, system_prompt)



        assert messages[-1]["role"] == "assistant"
        assert len(messages[-1]["content"]) == 1
        solution = messages[-1]["content"][0]["text"]

        prompt_msg = messages[:-1]
        prompt = autotokenizer.apply_chat_template(
            prompt_msg,
            tokenize=False,
            add_generation_prompt=False
        )
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        prompt += assistant_text
        images: List[Image.Image] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            for part in msg["content"]:
                if part["type"] == "image":
                    img = encode_image_to_pil(part["image"])
                    images.append(img)
                elif part["type"] == "image_url":
                    img = encode_image_to_pil(part["image_url"]['url'])
                    images.append(img)
                else:
                    pass
        if not images:
            images = None

        inputs = {
            "prompt": prompt,
        }
        if images is not None:
            inputs["multi_modal_data"] = {"image": images}
        completion = client.completions.create(
            model=args.model_id,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        continuation = completion.choices[0].text
        response =  continuation.strip()
        print("Response:", response.replace("\n", " "))
        # 简单过滤
        #print("Response:", response.replace("\n", " "))
        if response.count("Action:") != 1:
            return None, extra_info
        if response.count("Motion:") > 1:
            return None, extra_info
        if response.count("Grounding:") > 1:
            return None, extra_info

        suffix = check_suffix(response)
        


        if suffix is None:
            return None, extra_info
        sa = tokenizer.decode(response)[0]
        gt = tokenizer.decode(solution)[0]

        if judge_equal_action(sa, gt):
            if filter_noop and "Action: move(0, 0) and press()" in solution and "Action: move(0, 0) and press() and" not in solution:
                return None, extra_info

            ex = deepcopy(example)
            ex["conversations"][-1]["content"] = [{"type": "text", "text": response}]
            return ex, {"suffix": suffix, "task": task}
        return None, extra_info

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing example {idx}: {e}")
        return None, extra_info

def process_one_diff_coa(example, idx, args, system_prompt, autotokenizer, tokenizer, client, filter_noop=True, statistic=True):
    """并行处理一条数据，返回 (processed_example, suffix) 或 (None, None)"""

    try:
        if len(example["image"]) == 0:
            return None, None

        # 注意：Ray 里的函数不能直接用外部对象，需要自己加载
        from openagents.agents.utils.system_prompt import mix_action_system_prompt, text_coa_system_prompt, grounding_coa_system_prompt, motion_coa_system_prompt
        from openagents.agents.utils.action_mapping import TextActionTokenizer
        from copy import deepcopy


        failed = 0
        returned = []
        for jdx, sp in tqdm(enumerate([text_coa_system_prompt, grounding_coa_system_prompt, motion_coa_system_prompt])):
            messages, images = parse_msg_mix_coa(example["conversations"], example["image"], args.only_raw_history, sp)
            assert messages[-1]["role"] == "assistant"
            assert len(messages[-1]["content"]) == 1
            solution = messages[-1]["content"][0]["text"]

            for i in tqdm(range(3)):
                completion = client.chat.completions.create(
                    model=args.model_id,
                    messages=messages,                     # ✅ 替换 prompt
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    extra_body={
                        "skip_special_tokens": False,
                        "top_k": args.top_k
                    }
                )
                breakpoint()
                continuation = completion.choices[0].message.content
                if jdx%3 == 1 and "Grounding:" not in continuation:
                    continue
                if jdx%3 == 2 and "Motion:" not in continuation:
                    continue
            
            if (jdx%3 == 1 and "Grounding:" not in continuation) or (jdx%3 == 2 and "Motion:" not in continuation):
                failed += 1
        
            response = continuation.strip()

            # 简单过滤
            #print("Response:", response.replace("\n", " "))
            if response.count("Action:") != 1:
                continue
            if response.count("Motion:") > 1:
                continue
            if response.count("Grounding:") > 1:
                continue
            sa = tokenizer.decode(response)[0]
            if filter_noop and "Action: move(0, 0) and press()" in response and "Action: move(0, 0) and press() and" not in response:
                print("skip static action:", response.replace("\n", " ").strip())

            gt = tokenizer.decode(solution)[0]
            
            if judge_equal_action(sa, gt) or args.keep_failed:
                ex = deepcopy(example)
                ex["conversations"][-1]["content"] = [{"type": "text", "text": response}]
                if "Grounding:" in response and "Motion:" in response:
                    suffix = "Grounding_Motion:"
                elif "Grounding:" in response:
                    suffix = "Grounding:"
                elif "Motion" in response:
                    suffix = "Motion:"
                else:
                    suffix = "Action:" 
                returned.append((ex, {"suffix": suffix}))

        return returned, failed

    except Exception as e:
        import traceback
        traceback.print_exc()
        breakpoint()
        print(f"Error processing example {idx}: {e}")
        return None, {}

#mix coa, statistic for success rate.
def process_one_given_as(example, idx, args, system_prompt, autotokenizer, tokenizer, client, filter_noop=True, statistic=True):
    """并行处理一条数据，返回 (processed_example, suffix) 或 (None, None)"""

    try:
        if len(example["image"]) == 0:
            return None, None

        # 注意：Ray 里的函数不能直接用外部对象，需要自己加载
        from openagents.agents.utils.system_prompt import mix_action_system_prompt
        from openagents.agents.utils.action_mapping import TextActionTokenizer
        from copy import deepcopy

        messages, images = parse_msg_mix_coa(example["conversations"], example["image"], args.only_raw_history, system_prompt)


        assert messages[-1]["role"] == "assistant"
        assert len(messages[-1]["content"]) == 1
        solution = messages[-1]["content"][0]["text"]

        for jdx, suffix in enumerate(["Motion:", "Action:", "Grounding:"]):
            prompt_msg = messages[:-1] + [
                {"role": "assistant", "content": [{"type": "text", "text": suffix}]}
            ]
            prompt = autotokenizer.apply_chat_template(
                prompt_msg,
                tokenize=False,
                add_generation_prompt=False
            )
            prompt = prompt + "<|im_start|>assistant\n" + suffix
            pil_images = [Image.open(p["image_path"]).convert("RGB") for p in images]
            breakpoint()
            completion = client.completions.create(
                model=args.model_id,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            continuation = completion.choices[0].text
            response = suffix + continuation.strip()

            # 简单过滤
            #print("Response:", response.replace("\n", " "))
            if response.count("Action:") != 1:
                continue
            if response.count("Motion:") > 1:
                continue
            if response.count("Grounding:") > 1:
                continue

            sa = tokenizer.decode(response)[0]
            if filter_noop and "Action: move(0, 0) and press()" in response and "Action: move(0, 0) and press() and" not in response:
                print("skip static action:", response.replace("\n", " ").strip())
                #continue
            gt = tokenizer.decode(solution)[0]
            
            if judge_equal_action(sa, gt) and not args.keep_failed:
                ex = deepcopy(example)
                ex["conversations"][-1]["content"] = [{"type": "text", "text": response}]
                return ex, {"suffix": suffix}

        assert 0
        return None, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        breakpoint()
        print(f"Error processing example {idx}: {e}")
        return None, {}





def process_one_given_as_online(example, idx, args, system_prompt, autotokenizer, tokenizer, client, filter_noop=True, statistic=True):
    """并行处理一条数据，返回 (processed_example, suffix) 或 (None, None)"""

    try:
        if len(example["image"]) == 0:
            return None, None

        # 注意：Ray 里的函数不能直接用外部对象，需要自己加载
        from openagents.agents.utils.system_prompt import mix_action_system_prompt
        from openagents.agents.utils.action_mapping import TextActionTokenizer
        from copy import deepcopy

        messages, images = parse_msg_mix_coa(example["conversations"], example["image"], args.only_raw_history, system_prompt)


        assert messages[-1]["role"] == "assistant"
        assert len(messages[-1]["content"]) == 1
        solution = messages[-1]["content"][0]["text"]

        returned = []
        for jdx, suffix in enumerate(["Grounding:","Motion:", "Action:" ]):
            prompt_msg = messages[:-1] + [
                {"role": "assistant", "content": [{"type": "text", "text": suffix}]}
            ]
            prompt = autotokenizer.apply_chat_template(
                prompt_msg,
                tokenize=False,
                add_generation_prompt=False
            )
            prompt = prompt + "<|im_start|>assistant\n" + suffix
            pil_images = [Image.open(p["image_path"]).convert("RGB") for p in images]

            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": images
                }
            }

            gen_kwargs = {}
            sampling = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k =  args.top_k,
                top_p = args.top_p,
                skip_special_tokens=False,
                **gen_kwargs,
            )

            for i in range(3):

                breakpoint()
                model_outputs = client.generate(
                    inputs,
                    sampling,
                )
                continuation = model_outputs[0].outputs[0].text
                if (jdx%3 == 1 and "Grounding:" not in continuation) or (jdx%3 == 2 and "Motion:" not in continuation):
                    continue
                break

            if (jdx%3 == 1 and "Grounding:" not in continuation) or (jdx%3 == 2 and "Motion:" not in continuation):
                failed += 1
            response = suffix + continuation.strip()

            # 简单过滤
            #print("Response:", response.replace("\n", " "))
            if response.count("Action:") != 1:
                continue
            if response.count("Motion:") > 1:
                continue
            if response.count("Grounding:") > 1:
                continue

            sa = tokenizer.decode(response)[0]
            if filter_noop and "Action: move(0, 0) and press()" in response and "Action: move(0, 0) and press() and" not in response:
                print("skip static action:", response.replace("\n", " ").strip())
                #continue
            gt = tokenizer.decode(solution)[0]
            
            if judge_equal_action(sa, gt) and not args.keep_failed:
                ex = deepcopy(example)
                ex["conversations"][-1]["content"] = [{"type": "text", "text": response}]
                returned.append((ex, {"suffix": suffix}))
        return returned, failed


    except Exception as e:
        import traceback
        traceback.print_exc()
        breakpoint()
        print(f"Error processing example {idx}: {e}")
        return None, {}


def get_process_func(func_name):
    if func_name == "given_as":
        return process_one_given_as
    elif func_name == "mix_coa":
        return process_one_mix_coa
    elif func_name == "diff_coa":
        return process_one_diff_coa
    elif func_name == "given_as_online":
        return process_one_given_as_online
    else:
        raise ValueError(f"Unknown process function: {func_name}")