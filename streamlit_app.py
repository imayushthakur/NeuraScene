import os
import streamlit as st
from PIL import Image
from src.text_to_image.dall_e_generator import DallEGenerator
from src.image_to_video.video_generator import VideoGenerator
from src.utils.helpers import image_to_base64, video_to_base64

# Initialize generators
dalle = DallEGenerator()
video_gen = VideoGenerator()

# App config
st.set_page_config(
    page_title="AI Video BG Generator",
    page_icon="🎬",
    layout="wide"
)

# Sidebar
st.sidebar.header("Configuration")
duration = st.sidebar.slider("Video Duration (seconds)", 5, 60, 10)
subject = st.sidebar.selectbox("Subject Focus", 
              ["Physics", "Math", "Computer Science", "General"])

# Main interface
st.title("AI Video Background Generator")
st.markdown("Transform text prompts into dynamic video backgrounds using AI")

# Input section
col1, col2 = st.columns([3, 2])
with col1:
    prompt = st.text_area("Enter your background prompt:", 
              height=150,
              value="God particles with laser beams and digital plexus loops")
    
    generate_btn = st.button("Generate Video")

# Output display
with col2:
    st.markdown("### Generated Preview")
    image_placeholder = st.empty()
    video_placeholder = st.empty()
    status = st.empty()

# Generation logic
if generate_btn:
    with st.spinner("Generating image..."):
        try:
            # Step 1: Generate image
            img_result = dalle.generate_image(prompt, subject)
            
            if not img_result["success"]:
                st.error(f"Image generation failed: {img_result['error']}")
                raise st.stop()
            
            image_placeholder.image(img_result["image"], 
                                  caption="Generated Background Image")
            
            # Step 2: Generate video
            status.info("Animating image...")
            video_result = video_gen.generate_video(img_result["image"], 
                                                   duration=duration)
            
            if not video_result["success"]:
                st.error(f"Video generation failed: {video_result['error']}")
                raise st.stop()
            
            # Display video
            video_path = "temp_output.mp4"
            video_result["video_clip"].write_videofile(video_path, 
                                                     codec="libx264",
                                                     fps=30)
            
            video_placeholder.video(video_path)
            status.success("Generation complete!")
            
            # Download buttons
            st.download_button(
                label="Download Video",
                data=open(video_path, "rb"),
                file_name="ai_background.mp4",
                mime="video/mp4"
            )
            
        except Exception as e:
            st.error(f"Error in generation pipeline: {str(e)}")
