import os
import time
import logging
import base64
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
from PIL import Image
import torch
import io
import random
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment() -> bool:
    """
    Verify the environment is correctly set up.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"CUDA available: {cuda_available}, Device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, falling back to CPU")
    
    # Check for OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return False
    
    return True

def image_to_base64(image: Union[Image.Image, np.ndarray], format: str = "PNG") -> str:
    """
    Convert an image to base64 string.
    
    Args:
        image: PIL Image or numpy array
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
    
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def video_to_base64(video_path: Path) -> str:
    """
    Convert a video file to base64 string.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Base64 encoded string
    """
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    base64_str = base64.b64encode(video_data).decode()
    ext = video_path.suffix.lower()[1:]  # Remove the dot from extension
    return f"data:video/{ext};base64,{base64_str}"

def generate_unique_id(length: int = 10) -> str:
    """
    Generate a unique ID.
    
    Args:
        length: Length of the ID
        
    Returns:
        Unique ID string
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def ensure_directory(directory_path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path to directory
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def create_temp_file(suffix: str = None) -> Path:
    """
    Create a temporary file.
    
    Args:
        suffix: File extension
        
    Returns:
        Path to temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(path)

def normalize_text_prompt(prompt: str) -> str:
    """
    Normalize a text prompt for consistent DALL-E results.
    
    Args:
        prompt: Original prompt
        
    Returns:
        Normalized prompt
    """
    # Remove extra whitespace
    prompt = " ".join(prompt.split())
    
    # Add standard elements if not present
    required_elements = ["god particles", "laser beams", "digital plexus"]
    for element in required_elements:
        if element.lower() not in prompt.lower():
            prompt += f" Include {element}."
    
    return prompt

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
