import base64
import requests
import cv2
import io
from typing import Union,Literal,Dict,List
from rich import console
from pathlib import Path
from PIL import Image
import numpy as np
import imageio
import av
import os

def encode_image_to_pil(image_input: Union[np.ndarray,str, Path,Image.Image ]):
    if isinstance(image_input, str) and (image_input[0]=='/' or image_input[:4] == "http"):
        try:
            img_data = base64.b64decode(image_input)
            img = Image.open(io.BytesIO(img_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
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
  
def encode_image_to_np(image_input: Union[np.ndarray, str, Path, Image.Image]) -> np.ndarray:
    """
    将图片输入统一转为 RGB 格式的 NumPy 数组 (H, W, 3)，dtype=uint8
    支持输入：
        - np.ndarray (假定是RGB或RGBA)
        - str / Path (本地路径)
        - base64 字符串 (自动解码)
        - PIL.Image.Image
    """
    # 1. base64 字符串（用是否是纯字符串且非路径来区分）
    if isinstance(image_input, str) and not Path(image_input).exists():
        try:
            img_data = base64.b64decode(image_input)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except Exception:
            raise ValueError("Invalid base64 image data or unsupported format.")

    # 2. 文件路径
    elif isinstance(image_input, (str, Path)):
        try:
            img = Image.open(image_input).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except IOError:
            raise ValueError("Could not open the image file. Check the path and file format.")

    # 3. NumPy 数组
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # 灰度
            return np.stack([image_input] * 3, axis=-1).astype(np.uint8)
        elif image_input.ndim == 3 and image_input.shape[2] in (3, 4):  # RGB 或 RGBA
            return image_input[:, :, :3].astype(np.uint8)
        else:
            raise ValueError("Unsupported ndarray shape for image conversion.")

    # 4. PIL 图片
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert('RGB'), dtype=np.uint8)

    else:
        raise TypeError("Unsupported image input type. Supported types: str, pathlib.Path, numpy.ndarray, PIL.Image.Image")
    
def encode_image_to_bytes(image_input, format='JPEG'):
    img = encode_image_to_pil(image_input)
    buffer = io.BytesIO()
    try:
        img.save(buffer, format=format)
    except IOError:
        raise ValueError("Could not save image to bytes. Check the image format and data.")
    return buffer.getvalue()

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

def image_crop_inventory(image):
    if type(image)==str:
        temp_image = cv2.imread(image)
        height,width = temp_image.shape[:2]
        assert(height==360 and width==640)
    elif type(image)==np.ndarray:
        temp_image = image
    else:
        raise Exception(f"image错误的类型{type(image)}")
    scene = temp_image[:320,:,:]
    hotbars = np.zeros((9,16,16,3),dtype=np.uint8)
    left,top,w,h = 230,357,16,16
    for i in range(9):
        hotbars[i]=temp_image[top-h:top,left+(2*i+1)*2+i*w:left+2*(2*i+1)+(i+1)*w,:]
    return scene,hotbars


class VideoProcessor:
    def __init__(self, video_path: Union[Path, str], verbose: bool = False):
        self.video_path = Path(video_path)
        self.video_suffix = self.video_path.suffix
        self.verbose = verbose
        self.my_console = console.Console()
        self.cap = None
        self.all_frames = None  # 存储所有帧（仅在 fast=True 时使用）
        self.is_open = False

    def __len__(self) -> int:
        """Return the total number of frames in the video."""
        if self.video_suffix == ".mp4":
            if self.all_frames is not None:
                return len(self.all_frames)  # 如果 fast=True，则返回缓存的帧数
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self.video_suffix == ".npy":
            return len(self.all_frames)
        else:
            raise Exception(f"Unsupported video format: {self.video_suffix}")

    def _check_video_exists(self) -> bool:
        """Check if the video file exists."""
        if not self.video_path.exists():
            if self.verbose:
                self.my_console.log(f"[red]Error: The video {self.video_path} does not exist.[/red]")
            return False
        if self.verbose:
            self.my_console.log(f"[green]Video {self.video_path} exists.[/green]")
        return True

    def open(self, fast: bool = True) -> bool:
        """Open the video, supporting fast loading (loading all frames at once)."""
        if self.is_open:
            return True
        if not self._check_video_exists():
            return False

        if self.video_suffix != ".mp4":
            raise Exception(f"Unsupported video format: {self.video_suffix}")


        if fast:
            # Load all frames at once using imageio (RGB format)
            vid = imageio.get_reader(self.video_path, format='ffmpeg')
            self.all_frames = np.array([frame for frame in vid], dtype=np.uint8)
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception(f"Error: Could not open the video: {self.video_path}")
            if self.verbose:
                max_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.my_console.log(f"Video frame length: {max_frames}")

        if self.verbose:
            self.my_console.log(f"Video {self.video_path} opened successfully.")

        return True

    def get_frame(self, idx: int) -> np.ndarray:
        """Retrieve a specific frame from the video."""
        if self.verbose:
            self.my_console.log(f"Getting frame {idx}...")

        if self.all_frames is not None:
            if not (0 <= idx < len(self.all_frames)):
                raise IndexError(f"Frame index {idx} out of range.")
            frame = self.all_frames[idx]
        elif self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                raise Exception(f"Error: Failed to read frame {idx} from video {self.video_path}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise Exception("Video is not opened yet.")

        return frame

    def get_all_frames(self):
        if self.all_frames is not None:
            return self.all_frames
        elif self.cap is not None:
           frames = []
           for idx in range(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception(f"Error: Failed to read frame {idx} from video {self.video_path}.")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.apped(frame)
        else:
            raise Exception("Video is not opened yet.")

        return frames 

    def get_frames(self, start_idx:int, end_idx:int)-> List[np.ndarray]:
        if self.all_frames is not None:
            if not (0 <= start_idx < end_idx):
                raise IndexError(f"Frame index {start_idx} out of range.")
            if not end_idx < len(self.all_frames):
                raise IndexError(f"Frame index {end_idx} out of range.")
            frames = self.all_frames[start_idx:end_idx]
        elif self.cap is not None:
            frames = []
            for idx in range(start_idx,end_idx):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception(f"Error: Failed to read frame {idx} from video {self.video_path}.")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.apped(frame)
        else:
            raise Exception("Video is not opened yet.")

        return frames

    def close(self) -> bool:
        """Release resources and close the video file."""
        if self.verbose:
            self.my_console.log(f"Closing video {self.video_path}...")

        if self.video_suffix == ".mp4":
            if self.cap is not None:
                self.cap.release()
                cv2.destroyAllWindows()
                if self.verbose:
                    self.my_console.log(f"Video {self.video_path} (MP4) has been closed.")
        elif self.video_suffix == ".npy":
            self.all_frames = None  # 释放 numpy 数组
            if self.verbose:
                self.my_console.log(f"Video {self.video_path} (NPY) frames have been released.")
        return True
    
def save_video(frames, video_file, fps=20, no_bframe:bool=False):
    if not frames:
        print(f"警告: 没有帧可以保存到 {video_file}。")
        return
    height, width = frames[0].shape[0], frames[0].shape[1]
    
    try:
        with av.open(video_file, mode="w", format='mp4') as container:
            stream = None
            if no_bframe:
                stream = container.add_stream("libx264", rate=fps)
            else:
                stream = container.add_stream("h264", rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            if no_bframe:
                stream.options = {
                    'bframes': '0',
                    'preset': 'medium'
                }
            for idx, frame_data in enumerate(frames):
                try:
                    frame_to_encode = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    for packet in stream.encode(frame_to_encode):
                        container.mux(packet)
                except Exception as e:
                    print(f"错误：在编码第 {idx} 帧时发生错误: {e}")
                    continue
            for packet in stream.encode():
                container.mux(packet)
        print(f"视频成功保存到: {video_file}")
    except Exception as e:
        print(f"保存视频 {video_file} 时失败: {e}")
        if os.path.exists(video_file):
            try:
                os.remove(video_file)
                print(f"已清理损坏文件: {video_file}")
            except Exception as oe:
                print(f"清理文件失败: {oe}")