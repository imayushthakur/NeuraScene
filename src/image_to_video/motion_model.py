import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from src.config import MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfAttention(nn.Module):
    """Self attention module for the motion model."""
    
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class MotionEncoder(nn.Module):
    """Encoder for motion representation."""
    
    def __init__(self, in_channels=3, dim=64, depth=4):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Conv2d(in_channels if i == 0 else dim, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                SelfAttention(dim),
                nn.MaxPool2d(2)
            ]))
            
        self.motion_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 8)
        )
        
    def forward(self, x):
        for conv, norm, attn, pool in self.layers:
            x = pool(F.gelu(norm(conv(x))))
            b, c, h, w = x.shape
            x = x + attn(x.reshape(b, c, h * w).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)
        
        return self.motion_latent(x)

class MotionDecoder(nn.Module):
    """Decoder for generating motion from latent representation."""
    
    def __init__(self, dim=64, out_channels=3, seq_length=30):
        super().__init__()
        
        self.seq_length = seq_length
        self.latent_to_seq = nn.Sequential(
            nn.Linear(dim * 8, dim * 4 * seq_length),
            nn.GELU(),
            nn.Unflatten(1, (seq_length, dim * 4))
        )
        
        self.up_layers = nn.ModuleList([])
        sizes = [dim * 4, dim * 2, dim, dim // 2]
        
        for i in range(len(sizes) - 1):
            self.up_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(sizes[i], sizes[i+1], 4, 2, 1),
                nn.GroupNorm(8, sizes[i+1]),
                SelfAttention(sizes[i+1])
            ]))
            
        self.final_conv = nn.Conv2d(sizes[-1], out_channels, 1)
        
    def forward(self, x):
        x = self.latent_to_seq(x)
        b, t, c = x.shape
        
        # Initial spatial dimensions
        h, w = 8, 8
        x = x.reshape(b * t, c // (h * w), h, w)
        
        for conv, norm, attn in self.up_layers:
            x = F.gelu(norm(conv(x)))
            b_t, c, h, w = x.shape
            x = x + attn(x.reshape(b_t, c, h * w).transpose(1, 2)).transpose(1, 2).reshape(b_t, c, h, w)
        
        x = self.final_conv(x)
        
        # Reshape back to batch, time, channels, height, width
        _, c, h, w = x.shape
        x = x.reshape(b, t, c, h, w)
        
        return x

class DecoupledMotionModel(nn.Module):
    """
    Self-supervised model for decoupled motion and appearance.
    Adds motion to static images.
    """
    
    def __init__(self, dim=64, seq_length=30):
        super().__init__()
        
        self.appearance_encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )
        
        self.motion_encoder = MotionEncoder(in_channels=3, dim=dim)
        self.motion_decoder = MotionDecoder(dim=dim, out_channels=dim, seq_length=seq_length)
        
        self.final_renderer = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.seq_length = seq_length
        
    def forward(self, image):
        b, c, h, w = image.shape
        
        # Extract appearance features
        appearance = self.appearance_encoder(image)
        
        # Generate motion sequence
        motion_latent = self.motion_encoder(image)
        motion_seq = self.motion_decoder(motion_latent)
        
        # Combine appearance and motion for each frame
        output_frames = []
        
        for t in range(self.seq_length):
            # Extract motion features for this timestep
            motion_t = motion_seq[:, t]
            
            # Expand appearance to match batch size
            appearance_expanded = appearance.expand(b, -1, -1, -1)
            
            # Concatenate appearance and motion features
            combined = torch.cat([appearance_expanded, motion_t], dim=1)
            
            # Generate frame
            frame = self.final_renderer(combined)
            output_frames.append(frame)
        
        # Stack frames along time dimension
        output_video = torch.stack(output_frames, dim=1)
        
        return output_video
    
    def generate_video(self, image, device="cuda", return_frames=False):
        """Generate a video from a static image"""
        self.eval()
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = self(image_tensor)
            
            if return_frames:
                # Return as numpy arrays
                frames = [frame.squeeze(0).permute(1, 2, 0).cpu().numpy() for frame in output[0]]
                return frames
            else:
                # Return as a single tensor [T, H, W, C]
                return output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()

class MotionModelManager:
    """Manager class for loading and using the motion model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the motion model manager."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = DecoupledMotionModel(dim=64, seq_length=30).to(self.device)
        
        # Load weights if model path is provided
        if model_path and model_path.exists():
            logger.info(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Try to find pretrained weights in the models directory
            default_model_path = MODELS_DIR / "motion_model.pth"
            if default_model_path.exists():
                logger.info(f"Loading model weights from {default_model_path}")
                self.model.load_state_dict(torch.load(default_model_path, map_location=self.device))
            else:
                logger.warning("No model weights found. Using randomly initialized weights.")
                logger.warning(f"Please download pretrained weights to {default_model_path}")
    
    def generate_video_from_image(self, image, fps=30, duration=10.0, return_tensor=False):
        """
        Generate a video from a static image.
        
        Args:
            image: PIL Image or tensor
            fps: Frames per second
            duration: Duration of video in seconds
            return_tensor: If True, return tensor instead of numpy array
            
        Returns:
            Numpy array or tensor of video frames
        """
        # Convert PIL image to tensor if needed
        if not isinstance(image, torch.Tensor):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            image = transform(image)
        
        # Number of frames to generate
        num_frames = int(fps * duration)
        seq_length = self.model.seq_length
        repeats = (num_frames + seq_length - 1) // seq_length
        
        all_frames = []
        for _ in range(repeats):
            frames = self.model.generate_video(image, device=self.device, return_frames=True)
            all_frames.extend(frames)
        
        # Trim to the desired number of frames
        all_frames = all_frames[:num_frames]
        
        if return_tensor:
            return torch.tensor(np.array(all_frames))
        else:
            return np.array(all_frames)
