 # app.py
import streamlit as st
from transformers import pipeline  # For Chat-like functionality
from diffusers import StableDiffusionPipeline  # For image generation
import os
import subprocess

# App Configuration
st.set_page_config(
    page_title="Super AI App",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Super AI App: Chat, Create Images, Generate Videos")

# Sidebar Options
option = st.sidebar.radio(
    "Select Functionality:",
    ("Chat with AI", "Create Images", "Generate Videos"),
)

# Chat with AI
if option == "Chat with AI":
    st.header("AI Chat Assistant")
    chat_input = st.text_input("Ask me anything:")
    if st.button("Send"):
        chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")
        response = chatbot(chat_input)
        st.write("AI:", response[0]["generated_text"])

# Create Images
elif option == "Create Images":
    st.header("AI-Powered Image Creation")
    prompt = st.text_input("Describe the image you want to create:")
    if st.button("Generate Image"):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.to("cuda")  # Use GPU if available
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image")

# Generate Videos
elif option == "Generate Videos":
    st.header("AI-Powered Video Generation")
    video_prompt = st.text_input("Describe the video content:")
    if st.button("Generate Video"):
        st.write("Generating video... This may take a while.")
        # Placeholder for integration with video generation API or FFMPEG
        video_path = "output_video.mp4"  # Example placeholder
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.error("Video generation failed. Try again.")

# UI/UX Enhancements
st.sidebar.markdown("---")
st.sidebar.info("Built with 💻 Streamlit and Free Open Source Tools")
