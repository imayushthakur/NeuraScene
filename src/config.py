import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "pretrained"
EXAMPLES_DIR = BASE_DIR / "examples"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(EXAMPLES_DIR / "inputs").mkdir(parents=True, exist_ok=True)
(EXAMPLES_DIR / "outputs").mkdir(parents=True, exist_ok=True)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# DALL-E configuration
DALLE_API_VERSION = "dall-e-3"
DALLE_SIZE = "1024x1024"
DALLE_QUALITY = "standard"
DALLE_STYLE = "vivid"

# Video generation configuration
VIDEO_DURATION = 10  # seconds
VIDEO_FPS = 30
VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 1024

# Templates for text prompts
TEXT_PROMPT_TEMPLATE = """
Create a visually stunning background image for educational content with {description}. 
Include god particles, laser beams, and digital plexus loops.
Make it suitable for teaching {subject} concepts.
Ensure the image is dynamic, high-resolution, and has a clean, professional aesthetic.
"""

# API Server configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_PREFIX = "/api/v1"
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"
