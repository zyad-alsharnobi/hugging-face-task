import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import requests
import time
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set page config
st.set_page_config(page_title="Image Analysis App", layout="wide")

# Initialize session state
if 'generated_image_path' not in st.session_state:
    st.session_state.generated_image_path = None

# Hugging Face API configuration
HF_TOKEN = "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"
BLIP_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
DETR_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_image(prompt):
    client = InferenceClient(
        "stabilityai/stable-diffusion-xl-base-1.0",
        token=HF_TOKEN
    )
    image = client.text_to_image(prompt)
    return image

def get_image_caption(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    
    retries = 5
    for _ in range(retries):
        response = requests.post(BLIP_URL, headers=headers, data=data)
        output = response.json()
        if 'error' not in output:
            return output[0]['generated_text']
        time.sleep(2)
    return "Unable to generate caption"

def detect_objects(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(DETR_URL, headers=headers, data=data)
    return response.json()

def visualize_detections(image_path, detections):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for idx, detection in enumerate(detections):
        box = detection['box']
        label = detection['label']
        score = detection['score']
        
        rect = patches.Rectangle(
            (box['xmin'], box['ymin']),
            box['xmax'] - box['xmin'],
            box['ymax'] - box['ymin'],
            linewidth=2,
            edgecolor=colors[idx % len(colors)],
            facecolor='none'
        )
        
        ax.add_patch(rect)
        plt.text(
            box['xmin'], box['ymin'] - 5,
            f'{label}: {score:.2f}',
            color=colors[idx % len(colors)],
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.axis('off')
    return fig

# Main app
st.title("Image Analysis App")

# Image generation section
st.header("1. Generate Image")
prompt = st.text_input("Enter prompt for image generation:", "dog and cat playing football")
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt)
        # Save the image
        image_path = "generated_image.jpg"
        image.save(image_path)
        st.session_state.generated_image_path = image_path
        st.image(image, caption="Generated Image", use_column_width=True)

# Image captioning section
st.header("2. Generate Description")
if st.button("Generate Caption"):
    if st.session_state.generated_image_path:
        with st.spinner("Generating caption..."):
            caption = get_image_caption(st.session_state.generated_image_path)
            st.write("Caption:", caption)
    else:
        st.warning("Please generate an image first!")

# Object detection section
st.header("3. Detect Objects")
if st.button("Detect Objects"):
    if st.session_state.generated_image_path:
        with st.spinner("Detecting objects..."):
            detections = detect_objects(st.session_state.generated_image_path)
            fig = visualize_detections(st.session_state.generated_image_path, detections)
            st.pyplot(fig)
    else:
        st.warning("Please generate an image first!")