import os
import time
import logging
from typing import Optional, Dict, Any
import requests
from pathlib import Path
import openai
from PIL import Image
from io import BytesIO

from src.config import OPENAI_API_KEY, DALLE_API_VERSION, DALLE_SIZE, DALLE_QUALITY, DALLE_STYLE, TEXT_PROMPT_TEMPLATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DallEGenerator:
    """Class for generating images from text using OpenAI's DALL-E model."""
    
    def __init__(self):
        """Initialize the DALL-E generator with API key."""
        openai.api_key = OPENAI_API_KEY
        self.model = DALLE_API_VERSION
        self.size = DALLE_SIZE
        self.quality = DALLE_QUALITY
        self.style = DALLE_STYLE
        logger.info(f"Initialized DALL-E generator with model: {self.model}")
    
    def format_prompt(self, text: str, subject: str = "general") -> str:
        """Format the input text into a prompt suitable for DALL-E."""
        return TEXT_PROMPT_TEMPLATE.format(description=text, subject=subject)
    
    async def generate_image(self, text: str, subject: str = "general", 
                       output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate an image from text using DALL-E.
        
        Args:
            text: The text description to generate an image from
            subject: The educational subject for context
            output_path: Optional path to save the image to
            
        Returns:
            Dict containing the image data and metadata
        """
        prompt = self.format_prompt(text, subject)
        logger.info(f"Generating image with prompt: {prompt[:100]}...")
        
        try:
            response = openai.Image.create(
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                style=self.style,
                n=1,
                response_format="url"
            )
            
            image_url = response['data'][0]['url']
            
            # Download the image
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            
            # Save the image if output path is provided
            if output_path:
                image.save(output_path)
                logger.info(f"Image saved to {output_path}")
            
            return {
                "success": True,
                "image": image,
                "url": image_url,
                "prompt": prompt,
                "metadata": {
                    "model": self.model,
                    "size": self.size,
                    "timestamp": time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def enhance_prompt_for_education(self, base_prompt: str) -> str:
        """Add educational context to the prompt."""
        enhancements = [
            "suitable for educational backgrounds",
            "clean and non-distracting",
            "with subtle motion elements",
            "inspiring creative thinking",
            "with abstract representations of knowledge"
        ]
        
        return f"{base_prompt}. Make it {', '.join(enhancements)}."
