from PIL import Image 
import numpy as np 
import io  
import base64 
from typing import Dict, Any, Literal, List, Union
from pathlib import Path
import requests
from io import BytesIO

    

def encode_obs(obs): 
    '''
    obs: numpy.array (info['pov'])
    '''
    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(np.uint8(obs))
    # 创建一个内存中的字节流
    buffer = io.BytesIO()
    # 将图像保存为 PNG 格式到字节流中
    image.save(buffer, format="PNG")
    # 获取字节流中的数据
    img_bytes = buffer.getvalue()
    # 对字节数据进行 Base64 编码
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def encode_image_base64(image: np.array) -> str:
    # Convert the image to a base64 string
    img = Image.fromarray(image)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    return img_base64

def encode_image_to_base64(image:Union[str,Path,Image.Image,np.ndarray], format='jpeg') -> str:
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
        image = Path(image)
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
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Case 4: If the input is a PIL.Image
    elif isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Raise an error if the input type is unsupported
    else:
        raise ValueError("Unsupported input type. Must be a URL (str), numpy array, or PIL.Image.")

def decode_obs(img_base64):
    '''
    将Base64编码的图像字符串解码为NumPy数组
    
    参数:
    img_base64: 图像的Base64编码字符串
    
    返回:
    numpy.array: 解码后的图像数组，格式与原始输入一致
    '''
    # 将Base64字符串解码为字节数据
    img_bytes = base64.b64decode(img_base64)
    
    # 将字节数据转换为内存中的文件对象
    buffer = io.BytesIO(img_bytes)
    
    # 使用PIL打开图像
    image = Image.open(buffer)
    
    # 将PIL图像转换为NumPy数组
    obs = np.array(image)
    
    return obs

import xml.etree.ElementTree as ET
import re
from sam2.build_sam import build_sam2_camera_predictor
import numpy as np
import os
import torch
import cv2

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

SEGMENT_MAPPING = {
    "Kill": 0, 
    "Use": 3, 
    "Mine": 2, 
    "Interact": 3, 
    "Craft": 4, 
    "move_camera": 4,
    "Switch": 5, 
    "Approach": 6, 
    "Explore": -1,
    "move_camera": -1,
}

class SAMSession:
    def __init__(self, sam_path: str, segment_type: str = "Approach"):
        start_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.sam_path = sam_path
        self.clear_points()
        
        self.sam_choice = 'base'
        self.load_sam()
        
        self.tracking_flag = True
        self.points = []
        self.points_label = []
        self.able_to_track = False
        self.segment_type = segment_type # "Approach"
        self.obj_mask = np.zeros((224, 224), dtype=np.uint8)
        self.calling_rocket = False
        self.num_steps = 0

    def clear_points(self):
        self.points = []
        self.points_label = []
    
    def clear_obj_mask(self):
        self.obj_mask = np.zeros((224, 224), dtype=np.uint8)

    def load_sam(self):    
        ckpt_mapping = {
            'large': [os.path.join(self.sam_path, "sam2_hiera_large.pt"), "sam2_hiera_l.yaml"],
            'base': [os.path.join(self.sam_path, "sam2_hiera_base_plus.pt"), "sam2_hiera_b+.yaml"],
            'small': [os.path.join(self.sam_path, "sam2_hiera_small.pt"), "sam2_hiera_s.yaml"], 
            'tiny': [os.path.join(self.sam_path, "sam2_hiera_tiny.pt"), "sam2_hiera_t.yaml"]
        }
        sam_ckpt, model_cfg = ckpt_mapping[self.sam_choice]
        # first realease the old predictor
        if hasattr(self, "predictor"):
            del self.predictor
        self.predictor = build_sam2_camera_predictor(model_cfg, sam_ckpt)
        print(f"Successfully loaded SAM2 from {sam_ckpt}")
        self.able_to_track = False

    def segment(self, info, obs):
        if len(self.points) > 0 and len(self.points_label) > 0:
            self.able_to_track = True
            self.predictor.load_first_frame(info["pov"])
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, 
                obj_id=0,
                points=self.points,
                labels=self.points_label,
            )
        else:
            out_obj_ids, out_mask_logits = self.predictor.track(info["pov"])
        self.obj_mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy() # 360, 640
        self.clear_points()
        return self.obj_mask
    
    def get_segment(self, obs, info, points = None, segment_type = None):
        if points is not None:
            self.points = [points[0]]
            self.points_label = [1]

        self.segment(info, obs)
        # import time
        # self.save_mask_image(info["pov"], self.obj_mask, f"./output/{time.time()}.png") # 将mask image保存以查看是否正确
        resize_shape = obs["image"].shape[:2]
        if segment_type is not None:
            obj_id = torch.tensor( SEGMENT_MAPPING[segment_type])
        else:
            obj_id = torch.tensor( SEGMENT_MAPPING[self.segment_type])
        obj_mask = self.obj_mask.astype(np.uint8)
        obj_mask = cv2.resize(obj_mask, resize_shape, interpolation=cv2.INTER_NEAREST)
        obj_mask = torch.tensor(obj_mask, dtype=torch.uint8)
        return {
            'segment': {
                'obj_id': obj_id, 
                'obj_mask': obj_mask, 
            }
        }
    
    def save_mask_image(self, image, obj_mask, path):
        image = image.copy()
        color = COLORS[ SEGMENT_MAPPING[self.segment_type] ]
        color = np.array(color).reshape(1, 1, 3)[:, :, ::-1]
        obj_mask = (obj_mask[..., None] * color).astype(np.uint8)
        image = cv2.addWeighted(image, 1.0, obj_mask, 0.5, 0.0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
