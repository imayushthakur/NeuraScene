import os
import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
from PIL import Image
import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from src.config import VIDEO_FPS, VIDEO_DURATION, VIDEO_WIDTH, VIDEO_HEIGHT
from src.image_to_video.motion_model import MotionModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerator:
    """Class for generating videos from static images."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the video generator."""
        self.motion_model = MotionModelManager(model_path)
        self.fps = VIDEO_FPS
        logger.info(f"Initialized VideoGenerator with FPS: {self.fps}")
    
    def generate_video(self, image: Image.Image, duration: float = VIDEO_DURATION, 
                       output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a video from a static image.
        
        Args:
            image: The PIL Image to animate
            duration: Duration of video in seconds
            output_path: Optional path to save the video
            
        Returns:
            Dict containing video data and metadata
        """
        start_time = time.time()
        logger.info(f"Generating {duration}s video at {self.fps} FPS...")
        
        try:
            # Resize image if needed
            if image.size != (VIDEO_WIDTH, VIDEO_HEIGHT):
                image = image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
            
            # Generate video frames
            frames = self.motion_model.generate_video_from_image(
                image, 
                fps=self.fps, 
                duration=duration
            )
            
            # Convert to uint8 format for moviepy
            frames = (frames * 255).astype(np.uint8)
            
            # Create video clip
            clip = mpy.ImageSequenceClip(list(frames), fps=self.fps)
            
            # Save video if output path is provided
            if output_path:
                clip.write_videofile(
                    str(output_path),
                    codec="libx264",
                    audio=False,
                    fps=self.fps,
                    verbose=False,
                    logger=None
                )
                logger.info(f"Video saved to {output_path}")
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "video_clip": clip,
                "frames": frames,
                "duration": duration,
                "fps": self.fps,
                "frame_count": len(frames),
                "processing_time": processing_time,
                "metadata": {
                    "original_image_size": image.size,
                    "video_size": (VIDEO_WIDTH, VIDEO_HEIGHT),
                    "timestamp": time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_educational_enhancements(self, frames: np.ndarray) -> np.ndarray:
        """
        Add educational enhancements to video frames.
        
        Args:
            frames: Numpy array of video frames [T, H, W, C]
            
        Returns:
            Enhanced frames
        """
        # Example enhancement: add subtle pulsing effect
        enhanced_frames = frames.copy()
        frame_count = len(frames)
        
        # Generate pulsing effect
        pulse_factor = 0.05  # Intensity of the pulse
        for i in range(frame_count):
            # Sinusoidal pulsing
            pulse = 1.0 + pulse_factor * np.sin(2 * np.pi * i / (frame_count / 3))
            enhanced_frames[i] = np.clip(frames[i] * pulse, 0, 1)
        
        return enhanced_frames
    
    def optimize_for_streaming(self, output_path: Path, target_size_mb: float = 10.0) -> Path:
        """
        Optimize video for streaming by controlling file size.
        
        Args:
            output_path: Path to the original video
            target_size_mb: Target size in megabytes
            
        Returns:
            Path to the optimized video
        """
        optimized_path = output_path.with_name(f"{output_path.stem}_optimized{output_path.suffix}")
        
        # Get original file size
        original_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        if original_size_mb <= target_size_mb:
            logger.info(f"Video already smaller than target size ({original_size_mb:.2f}MB <= {target_size_mb}MB)")
            return output_path
        
        # Calculate bitrate based on target size
        duration = VIDEO_DURATION
        target_bitrate = int((target_size_mb * 8 * 1024) / duration)
        
        # Create optimized video
        clip = mpy.VideoFileClip(str(output_path))
        clip.write_videofile(
            str(optimized_path),
            codec="libx264",
            audio=False,
            fps=self.fps,
            bitrate=f"{target_bitrate}k",
            verbose=False,
            logger=None
        )
        
        new_size_mb = os.path.getsize(optimized_path) / (1024 * 1024)
        logger.info(f"Optimized video: {original_size_mb:.2f}MB → {new_size_mb:.2f}MB")
        
        return optimized_path
