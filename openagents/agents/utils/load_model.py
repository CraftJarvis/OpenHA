'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-05 10:56:23
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-06-23 11:56:58
'''
from typing import Optional

ANTHROPIC_MODEL_LIST = {
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-haiku-20240307-vertex",
    "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229-vertex",
    "claude-3-5-sonnet"
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
}

OPENAI_MODEL_LIST = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
}
DEEPSEEK_MODEL_LIST = {
    "deepseek-chat", #0.1元/百万tokens 2元/百万tokens 4K token
    "deepseek-reasoner",
}

PROPRIETART_MODEL_LIST =set()
PROPRIETART_MODEL_LIST.update(DEEPSEEK_MODEL_LIST,OPENAI_MODEL_LIST,ANTHROPIC_MODEL_LIST)

def load_visual_model(checkpoint_path ="",LLM_backbone:Optional[str]=None,VLM_backbone:Optional[str]=None,**kwargs):
    if not checkpoint_path:
        raise AssertionError("checkpoint_path is required")
    
    checkpoint_path = checkpoint_path.lower().replace('-','_').replace("qwen2vl","qwen2_vl").replace("qwen2.5vl","qwen2.5_vl")
    
    if LLM_backbone is not None:
        pass
    elif "mistral" in checkpoint_path:
        LLM_backbone = "mistral"
    elif "vicuna" in checkpoint_path:
        LLM_backbone = "llama-2"
    elif "llama_3" in checkpoint_path or "llama3" in  checkpoint_path:
        LLM_backbone = "llama-3"
    elif "qwen2_vl" in checkpoint_path:
        LLM_backbone = "qwen2_vl"
    elif "qwen2.5_vl" in checkpoint_path:
        LLM_backbone = "qwen2.5_vl"
       
    if VLM_backbone is not None: 
        pass
    elif 'llava_next' in checkpoint_path or 'llava_v1.6'  in checkpoint_path:
        VLM_backbone = "llava-next"
    elif "qwen2_vl" in checkpoint_path:
        VLM_backbone = "qwen2_vl"
    elif "qwen2.5_vl" in checkpoint_path:
        VLM_backbone = "qwen2.5_vl"
    else:
        raise AssertionError

    return LLM_backbone,VLM_backbone
        