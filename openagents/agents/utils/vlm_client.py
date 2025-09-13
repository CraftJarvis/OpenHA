# --- standard library ---------------------------------------------------------
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import random
import traceback
from rich import print
import copy
import io
import math
import base64

from rich.console import Console
from rich.logging import RichHandler
import logging

from openai import OpenAI
from anthropic import Anthropic
import torch
from vllm import LLM, SamplingParams

from transformers import (
    AutoTokenizer,
    AutoModelForImageTextToText,
    Qwen2VLProcessor,
    Qwen2_5_VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from PIL import Image  
import numpy as np 
import cv2  
import requests  
from io import BytesIO 
from openagents.utils import img_utils

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

class VLMClient(object):
    """Unified client wrapper for online/offline VLM back‑ends.
    """

    _TP_SIZE_DEFAULT: int = torch.cuda.device_count() or 1
    _MAX_MODEL_LEN: int = 12288
    _MAX_NUM_SEQS: int = 1
    _GPU_MEM_UTIL: float = 0.40

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: int= 4096,
        top_logprobs: Optional[int] = None,
        get_sampled_logprobs: Optional[bool] = False,
        extra_body: Optional[dict]= None,
        model_id:Optional[str] = None,
        LLM_backbone: Optional[str] = None, 
        VLM_backbone: Optional[str] = None,
        limit_mm_per_prompt_image : Optional[int] = 5,
        limit_mm_per_prompt_video : Optional[int] = 1,
        if_token_ids: bool = False,
        tokenizer_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: Literal["online","openai","anthropic", "vllm", "hf", ] = "online",
        **kwargs: Any,
    ) -> None:
        
        # logger
        
        self.console: Console = Console()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)],
        )
        self.logger = logging.getLogger(__name__)
        
        # --- public attrs -------------------------------------------------
        self.mode =  mode
        self.temperature = temperature
        self.top_p = top_p if top_p is not None else 1.0
        self.top_k = top_k if top_k is not None else -1
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.get_sampled_logprobs = get_sampled_logprobs
        self.extra_body = {} if extra_body is None else extra_body
        self.model_path = str(model_path) if model_path is not None else model_path
        self.__api_key = api_key if api_key is not None else "EMPTY"
        self.base_url = base_url
        self.model_id = model_id
        self.limit_mm_per_prompt_image = limit_mm_per_prompt_image
        self.limit_mm_per_prompt_video = limit_mm_per_prompt_video

        # --- toolkits -----------------------------------------------------
        self.LLM_backbone,self.VLM_backbone = load_visual_model(checkpoint_path=model_path,LLM_backbone=LLM_backbone,VLM_backbone=VLM_backbone)
        self.tokenizer = None
        self.if_token_ids = if_token_ids
        if if_token_ids:
            self.tokenizer_path = tokenizer_path if tokenizer_path is not None else self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True) 
        self.processor_wrapper = None  
        self.set_processor_wrapper(model_name=self.VLM_backbone)

        if self.mode == "online" and model_id in ANTHROPIC_MODEL_LIST:
            self.mode = "anthropic"
                
        if self.mode in {"openai","online","online_lmdeploy"}:
            self._init_online_openai_backend()
        elif self.mode == "anthropic":
            self._init_online_anthropic_backend()
        elif self.mode == "vllm":
            self._init_offline_vllm_backend()
        elif self.mode == "hf": # "hf"
            self._init_hf_backend()
        else:
            raise ValueError("wrong mode: ", self.mode)
        
        self.logger.info(f"[VLMClient] initialized in '{self.mode}' mode")

    # ------------------------------------------------------------------
    # initialization helpers
    # ------------------------------------------------------------------
    def _init_online_openai_backend(self) -> None:
        """Initialise OpenAI compatible HTTP backend."""
        if self.mode == "online":
            self.extra_body["top_k"] = self.top_k
        self.extra_body["skip_special_tokens"] = False
        self.client = OpenAI(api_key=self.__api_key, base_url=self.base_url)
        if not self.model_id:
            models = self.client.models.list()
            self.model_id = models.data[0].id  # choose first available
        self.logger.debug(f"[OpenAI] Using model: {self.model_id}, base_url: {self.base_url}")
        
    def _init_online_anthropic_backend(self) -> None:
        """Initialise Anthropic compatible HTTP backend."""
        self.client = Anthropic(api_key=self.__api_key, base_url=self.base_url)
        if not self.model_id:
            models = self.client.models.list()
            self.model_id = models.data[0].id  # choose first available
        self.logger.debug(f"[Anthropic] Using model: {self.model_id}")

    def _init_offline_vllm_backend(self) -> None:
        """Initialise local vLLM engine."""
        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self._TP_SIZE_DEFAULT,
            max_model_len=self._MAX_MODEL_LEN,
            max_num_seqs=self._MAX_NUM_SEQS,
            gpu_memory_utilization=self._GPU_MEM_UTIL,
            trust_remote_code=True,
            limit_mm_per_prompt={"image":self.limit_mm_per_prompt_image, "video": self.limit_mm_per_prompt_video},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
        )
        if  self.VLM_backbone == "qwen2_vl":
            self.processor = Qwen2VLProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                do_rescale=False,
                patch_size=14,
                vision_feature_select_strategy="default",
            )
        elif self.VLM_backbone == "qwen2.5_vl":
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                do_rescale=False,
                patch_size=14,
                vision_feature_select_strategy="default",
            )

    def _init_hf_backend(self) -> None:
        """Initialise HuggingFace transformers model backend."""
        processor_cfg: Dict[str, Any] = {
            "patch_size": 14,
            "vision_feature_select_strategy": "default",
        }
        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.float16}
        if self.VLM_backbone == "qwen2_vl":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            processor_cfg.update({
                "min_pixels": 224 * 224,
                "max_pixels": 2048 * 2048,
            })
            self.processor = Qwen2VLProcessor.from_pretrained(self.model_path, trust_remote_code=True, **processor_cfg)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cuda:0",
                **model_kwargs,
            ).eval()
        elif self.VLM_backbone == "qwen2.5_vl":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            processor_cfg.update({
                "min_pixels": 224 * 224,
                "max_pixels": 2048 * 2048,
            })
            self.processor = Qwen2_5_VLProcessor.from_pretrained(self.model_path, trust_remote_code=True,  **processor_cfg)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cuda:0",
                **model_kwargs,
            ).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cuda:0",
                **model_kwargs,
            ).eval()

    def reset(self) -> None:
        """Refresh HTTP client (token expiry, etc.)."""
        if self.mode in {"openai","online"}:
            self._init_online_openai_backend()
            self.logger.debug("[VLMClient] HTTP client reset")
        elif self.mode == "anthropic":
            self._init_online_anthropic_backend()
            self.logger.debug("[VLMClient] HTTP client reset")

    def set_processor_wrapper(self, model_name: Optional[str] = None) -> None:
        model_name = model_name or self.model_id
        self.processor_wrapper = ProcessorWrapper(model_name=model_name, mode = self.mode)

    def generate(self, messages: List[dict[str, Any]], *, verbose: bool = False, if_token_ids: bool = False, if_token_record: bool = True) -> Dict[str, Any]:
        """Generate response.

        Returns
        -------
        dict with keys ``content``, ``action`` and optionally ``logprob``.
        """
        if verbose:
            self.logger.setLevel(logging.DEBUG)

        if_token_ids = if_token_ids or self.if_token_ids
        verbose_messages = copy.deepcopy(messages)
        for message in verbose_messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    content["image_url"] = ""
                    
        self.logger.debug(f"[generate/{self.mode}] {verbose_messages}")
        if self.mode in {"openai","online",}:
            outputs = self._generate_openai(messages, if_token_ids,if_token_record=if_token_record)
        elif self.mode == "anthropic":
            outputs = self._generate_anthropic(messages, if_token_ids,if_token_record=if_token_record)
        elif self.mode == "vllm":
            outputs =  self._generate_offline_vllm(messages, if_token_ids)
        else:
            outputs = self._generate_local_hf(messages, if_token_ids)
        self.logger.debug(f"[generate/{self.mode}] done -------------- ")
        return outputs
    
    def _generate_openai(self, messages: List[dict[str, Any]], if_token_ids: bool= False,if_token_record:bool= False,) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {}
        
        if self.top_logprobs:
            gen_kwargs.update({"top_logprobs": self.top_logprobs, "logprobs": True})
        elif self.get_sampled_logprobs:
            gen_kwargs.update({"logprobs": True})
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p = self.top_p,
                extra_body=self.extra_body,
                **gen_kwargs,
            )
            content_choice = chat_completion.choices[0]
            if if_token_record:
                token = {
                    "input" : chat_completion.usage.prompt_tokens,
                    "output" : chat_completion.usage.completion_tokens,
                }
        except Exception as e:
            print(chat_completion)
            traceback.print_exc() 
            raise Exception(f"error: {e}")
        try:
            content = content_choice.message.content
            action = self._maybe_tokenize(content, if_token_ids)
            outputs: Dict[str, Any] = {"content": content, "action": action}
            if self.top_logprobs:
                outputs["logprob"] = [
                    [(prob.token, prob.logprob) for prob in logprob.top_logprobs]
                    for logprob in content_choice.logprobs.content
                ]
                outputs["response_tokens"] = [[tokenlogprob.token, tokenlogprob.logprob] for tokenlogprob in content_choice.logprobs.content]
            if self.get_sampled_logprobs:
                outputs["sampled_logprobs"] = [
                    [logprob.token, logprob.logprob] for logprob in content_choice.logprobs.content
                ]
            if if_token_record:
                outputs["token"] = token
            if hasattr(content_choice.message,"reasoning_content"):
                outputs["reasoning_content"] = content_choice.message.reasoning_content
                outputs["all"] = content_choice
        except Exception as e:
            print(e)
            traceback.print_exc() 
            raise Exception(f"error: {e}")
        return outputs
    
    def _generate_anthropic(self, messages: List[Dict[str, Any]], if_token_ids: bool= False,if_token_record:bool= False,) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = copy.copy(self.extra_body)
        if "thinking" in self.extra_body:
            temperature = 1
        if messages[0].get("role","") == "system":
            extra_body["system"] = messages[0]["content"]
            messages = messages[1:]
        try:
            chat_completion = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
                **extra_body,
            )
        except Exception as e:
            print(chat_completion)
            raise Exception(f"error: {e}")
        outputs: Dict[str, Any] = {}
        for block in chat_completion.content:
            if block.type == "thinking":
                outputs["reasoning_content"] = block.thinking
            elif block.type == "text":
                content = block.text
                outputs["content"] = content
                outputs["action"] = self._maybe_tokenize(content, if_token_ids)
        if if_token_record:
            outputs["token"] = {
                "input" : chat_completion.usage.input_tokens,
                "output": chat_completion.usage.output_tokens,
            }
        return outputs

    def _generate_offline_vllm(self, messages: List[dict[str, Any]], if_token_ids: bool) -> Dict[str, Any]:

        gen_kwargs: Dict[str, Any] = {}
        if self.top_logprobs:
            gen_kwargs.update({"top_logprobs": self.top_logprobs, "logprobs": True})

        sampling = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k =  self.top_k,
            top_p = self.top_p,
            skip_special_tokens=False,
            **gen_kwargs,
        )
        
        assistant_text = ""
        while messages and messages[-1]["role"] == "assistant":
            for content in messages[-1]["content"]:
                assistant_text += content["text"]
            messages = messages[:-1]
            
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
            
        model_outputs = self.model.generate(
            inputs,
            sampling_params=sampling,
        )
        output_text = assistant_text + model_outputs[0].outputs[0].text
        #output_text = self.model.chat(messages, sampling_params=sampling, chat_template=self.processor.chat_template)[0].outputs[0].text

        outputs: Dict[str, Any] = {
            "content": output_text,
            "action": self._maybe_tokenize(output_text, if_token_ids),
        }
        self.logger.debug("[generate/vllm] done")
        return outputs

    def _generate_local_hf(self, messages: List[dict[str, Any]], if_token_ids: bool) -> Dict[str, Any]:
        images: List[Image.Image] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            for part in msg["content"]:
                if part["type"] == "image":
                    img = encode_image_to_pil(part["image"])
                    part["image"] = img
                    images.append(img)
        if not images:
            images = None
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=images, padding=True, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_cfg = {
            "do_sample": bool(self.temperature),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": 1.0,
        }
        if self.top_k > 0:
            gen_cfg["top_k"] = self.top_k
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens - input_len,
            **gen_cfg,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )[0][input_len:]
        content = self.processor.batch_decode([output_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        outputs: Dict[str, Any] = {"content": content, "action": output_ids if if_token_ids else content}
        self.logger.debug("[generate/hf] done")
        return outputs

    def _maybe_tokenize(self, text: str, flag: bool) -> Union[str, List[int]]:
        if not flag:
            return copy.deepcopy(text)
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialised, cannot return token ids")
        return self.tokenizer(text)["input_ids"]


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int,max_ratio:int
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(image: Image.Image,  factor: int, min_pixels: int, max_pixels: int,max_ratio:int) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_ratio=max_ratio,
        )
    image = image.resize((resized_width, resized_height))
    return image

def pil2base64(image):
    """强制中间结果为jpeg""" 
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def encode_image_to_pil(image_input):
    if isinstance(image_input, str) and image_input[:len("data:image")] == "data:image":
        try:
            base64_str = image_input.split(",")[1]
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except Exception:
            raise ValueError("Invalid base64 image data or unsupported format.")
    elif isinstance(image_input, (str, Path)): 
        try:
            img = Image.open(image_input)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except IOError:
            raise ValueError("Could not open the image file. Check the path and file format.")
    elif isinstance(image_input, np.ndarray):
        try:
            img = Image.fromarray(image_input)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except TypeError:
            raise ValueError("Numpy array is not in an appropriate format to convert to an image.")
    elif isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        return image_input
    else:
        raise TypeError("Unsupported image input type. Supported types are str, pathlib.Path, numpy.ndarray, and PIL.Image.")
    

def encode_image_to_base64(image:Union[str,Path,Image.Image,np.ndarray], format='JPEG') -> str:
    """Encode an image to base64 format, supports URL, numpy array, and PIL.Image."""

    # Case 1: If the input is a URL (str)
    image_encode = None
    if isinstance(image, str) and image[:4]=="http":
        try:
            response = requests.get(image)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to retrieve the image from the URL: {e}")
    elif isinstance(image, str) and image[0]=='/':
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    elif isinstance(image,Path):
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    # Case 3: If the input is a numpy array
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Case 4: If the input is a PIL.Image
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Raise an error if the input type is unsupported
    else:
        raise ValueError("Unsupported input type. Must be a URL (str), numpy array, or PIL.Image.")

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

def translate_cv2(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        # Convert PIL Image to NumPy array (PIL is in RGB)
        img_array = np.array(image)
        cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Check if the NumPy array is in RGB format and has three channels
        if image.shape[2] == 3:  # Only for color images
            cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            cv2_image = image  # No conversion needed for grayscale images
    elif isinstance(image, (str, Path)):
        # Read the image using cv2 (assumes BGR format)
        cv2_image = cv2.imread(str(image))  # Convert PosixPath to string if necessary
        if cv2_image is None:
            raise ValueError(f"The image path is incorrect or the file is not accessible: {image}")
    else:
        raise ValueError("Unsupported image format or path type")
    
    return cv2_image

    

class ProcessorWrapper:
    def __init__(self, model_name= "qwen2_vl",mode="online"):
        self.mode = mode
        self.model_name = model_name.replace("-","_")
        self.image_factor = 28
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = 1024 * 28 * 28  #16384 * 28 * 28
        self.max_ratio = 200

    def get_image_message(self,source_data:Union[str,Path,np.ndarray,Image.Image]):
        image_suffix = get_suffix(source_data)
        image = None
        if self.mode == "hf":
            image_message = {
                "type": "image",
                "image": source_data,
            }
        elif self.mode == "anthropic":
            image_suffix = "jpeg" if image_suffix=="jpg" else image_suffix
            image_message = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image_suffix}",
                        "data": encode_image_to_base64(source_data)
                    },
                }
        else:
            image = { "url": f"data:image/{image_suffix};base64,{encode_image_to_base64(source_data)}"}
            image_message = {
                    "type": "image_url",
                    "image_url": image,
                }
        return image_message

    def create_system_prompt(self,system_prompt:str=""):
        message = {
            "role":"system",
            "content": f"{system_prompt}\n",
        }
        return message

    def create_message_vllm(self,
                            role:Literal["user","assistant"]="user",
                            input_type:Literal["image","text"]="image",
                            image:Union[list,str,Path,np.ndarray,Image.Image]=None,
                            prompt:Union[list,str]="",):
        if role not in {"user","assistant"}:
            raise ValueError(f"a invalid role {role}")
        if isinstance(prompt,str):
            prompt = [prompt]
        message = {
            "role": role,
            "content": [],
        }
        if input_type=="image":
            if not isinstance(image,list):
                image = [image]
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}"
                })
                if idx < len(image):
                    message["content"].append(self.get_image_message(image[idx]))
            for idx in range(len(prompt), len(image)):
                message["content"].append(self.get_image_message(image[idx])) 
        else:
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}"
                })
        return message
    
    def create_text_input(self,conversations:list):
        text_prompt = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
        return text_prompt
    
    def create_image_input(self,image_input:Union[np.ndarray,str, Path,Image.Image ]):
        image = img_utils.encode_image_to_pil(image_input)
        # image = image_pixels
        # if image_path:
        #     image = Image.open(image_path)
        # if not isinstance(image, Image.Image):
        #     image = Image.fromarray(image.astype('uint8'))
        if "qwen2_vl" in self.model_name:
            image = fetch_image(image,factor=self.image_factor,min_pixels=self.min_pixels,max_pixels=self.max_pixels,max_ratio=self.max_ratio,)
        return image

def load_visual_model(checkpoint_path ="",LLM_backbone:Optional[str]=None,VLM_backbone:Optional[str]=None,**kwargs):
    if not checkpoint_path:
        return "",""
    
    checkpoint_path = checkpoint_path.lower().replace('-','_').replace("qwen2vl","qwen2_vl").replace("qwen2.5vl","qwen2.5_vl")
    
    if LLM_backbone is not None and LLM_backbone!="":
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
    else:
        LLM_backbone = ""
       
    if VLM_backbone is not None and VLM_backbone!="":
        pass
    elif 'llava_next' in checkpoint_path or 'llava_v1.6'  in checkpoint_path:
        VLM_backbone = "llava-next"
    elif "qwen2_vl" in checkpoint_path:
        VLM_backbone = "qwen2_vl"
    elif "qwen2.5_vl" in checkpoint_path:
        VLM_backbone = "qwen2.5_vl"
    else:
        VLM_backbone = ""

    return LLM_backbone,VLM_backbone
    
if __name__ == "__main__":
    import os
    from ultron.dataset.api import message_generate
    vlm_client = VLMClient(model_path="/share_data/limuyao/checkpoints/train/mc-motion-coa-qwen2-vl-7b-250809-A800-c32-e1-b8-a1/checkpoint-3850",if_token_ids=False,base_url="http://localhost:11000/v1",mode="online",get_sampled_logprobs=1)
    vlm_client.reset()
    messages = []
    messages.append(message_generate.create_system_message(system_prompt="\nYour role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <think> {thought with steps separated with '\n\n'} </think> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, Now, try to solve the following question through the above guidelines:\n",model_type="gpt-4o"))
    messages.append(message_generate.create_user_message(input_type="text",user_prompt="Give you nothing in the inventory, generate a step-by-step plan to obtain diamonds.",model_type="gpt-4o"))
    a = vlm_client.generate(messages=messages)
    print(a["sampled_logprobs"])