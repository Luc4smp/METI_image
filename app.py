# Streamlit Web Application for Handwritten Digit Generation
#
# Instructions:
# 1. Save this script as `app.py`.
# 2. Make sure you have your trained `cgan_generator.pth` file in the same directory.
# 3. Create a file named `requirements.txt` with the following content:
#    streamlit
#    torch
#    torchvision
# 4. Upload `app.py`, `cgan_generator.pth`, and `requirements.txt` to a GitHub repository.
# 5. Deploy the repository on Streamlit Community Cloud.

import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import numpy as np

# --- 1. Model and Environment Setup ---

# Use CPU for inference, as it's sufficient and widely available on deployment platforms.
device = torch.device("cpu")

# Define hyperparameters - these MUST match the ones used during training.
LATENT_DIM = 100
CHANNELS_IMG = 1
IMAGE_SIZE = 28
NUM_CLASSES = 10
EMBED_SIZE = 100

# --- 2. Re-define the Generator Model Architecture ---
# The model architecture must be identical to the one used for training.

class Generator(nn.Module):
    def __init__(self, latent_dim, channels_img, num_classes, embed_size):
        super(Generator, self).__init__()
        input_dim = latent_dim + embed_size
        self.embed = nn.Embedding(num_classes, embed_size)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, IMAGE_SIZE * IMAGE_SIZE * channels_img),
            nn.Tanh()
        )

    def forward(self, x, labels):
        embedding = self.embed(labels)
        x = torch.cat([x, embedding], dim=1)
        output = self.net(x)
        return output.view(-1, CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)

# --- 3. Function to Load Model and Generate Images ---

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model(model_path="cgan_generator.pth"):
    """Loads the trained generator model."""
    model = Generator(LATENT_DIM, CHANNELS_IMG, NUM_CLASSES, EMBED_SIZE).to(device)
    # Load the state dictionary. The map_location argument ensures that the model
    # loads correctly even if it was trained on a GPU and is now being run on a CPU.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def generate_images(model, digit, num_images=5):
    """Generates a specified number of images for a given digit."""
    with torch.no_grad():
        # Create latent noise vectors
        noise = torch.randn(num_images, LATENT_DIM, device=device)
        # Create labels for the desired digit
        labels = torch.LongTensor([digit] * num_images).to(device)
        # Generate images
        fake_images = model(noise, labels).cpu()
        # Denormalize the images from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2.0
    return fake_images

# --- 4. Streamlit User Interface ---

# Set up the page title and description
st.set_page_config(page_title="Digit Generator", layout="wide")
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained cGAN model.")

# Load the generator model
try:
    generator = load_model()

    # Create the user input section in the sidebar
    st.sidebar.header("Controls")
    selected_digit = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))
    generate_button = st.sidebar.button("Generate Images", type="primary")

    # A placeholder to show the generated images
    st.subheader(f"Generated images of digit {selected_digit}")

    if generate_button:
        with st.spinner(f"Generating images of digit {selected_digit}..."):
            # Generate the images
            images_tensor = generate_images(generator, selected_digit, num_images=5)
            
            # Display the images
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    # Convert tensor to numpy array for display
                    img_np = images_tensor[i].squeeze().numpy()
                    st.image(img_np, caption=f"Sample {i+1}", width=128, clamp=True)

    else:
        st.info("Select a digit and click 'Generate Images' to start.")

except FileNotFoundError:
    st.error("Model file 'cgan_generator.pth' not found.")
    st.write("Please make sure the trained model file is in the same directory as this script.")
except Exception as e:
    st.error(f"An error occurred: {e}")

