import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import pandas as pd
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import Siamese Model
from model.siamese_network import SiameseNetwork

st.set_page_config(page_title="Signature Verification System", layout="centered")

st.title("Signature Verification System")

# -----------------------------
# Selection Section
# -----------------------------

bank = st.radio(
    "Select Bank (Dataset)",
    ["Bank1", "Bank2", "Bank3"]
)

model_version = "Local Model"

# -----------------------------
# Sidebar & Threshold Settings
# -----------------------------

st.sidebar.header("Verification Settings")

# Default thresholds based on selected model
default_threshold = 1.4 if model_version == "Local Model" else 0.15

threshold = st.sidebar.slider(
    "Verification Threshold (Sensitivity)",
    min_value=0.05,
    max_value=2.0,
    value=default_threshold,
    step=0.05,
    help="Lower threshold = stricter verification. Recommended: 1.4 for Local, 0.15 for Global."
)

st.sidebar.markdown(f"**Current Threshold:** `{threshold:.2f}`")
st.sidebar.info("Tip: If genuine signatures are failing, increase the threshold. If forgeries are passing, decrease it.")

st.write(f"Verification Mode: **{bank}** using **{model_version}**")


# -----------------------------
# Load Model Dynamically
# -----------------------------

@st.cache_resource
def load_model(bank_name, version):
    
    filename = "local_model.pth" if version == "Local Model" else "global_model.pth"
    model_path = os.path.join(PROJECT_ROOT, bank_name, filename)

    if not os.path.exists(model_path):
        # Fallback for Global Model if not found in bank folder
        if version == "Global Model":
            model_path = os.path.join(PROJECT_ROOT, "server", "global_model.pth")
            if not os.path.exists(model_path):
                return None
        else:
            return None

    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model


model = load_model(bank, model_version)

# -----------------------------
# Load Dataset for Selected Bank
# -----------------------------

dataset_path = os.path.join(PROJECT_ROOT, bank, "pair_dataset.csv")

if os.path.exists(dataset_path):
    dataset_df = pd.read_csv(dataset_path)
else:
    dataset_df = None


# -----------------------------
# Reference Signature Finder (Multi-Ref)
# -----------------------------

def get_reference_signatures(account_id, num_refs=5):
    """Returns a list of up to num_refs genuine signature paths."""
    if dataset_df is None:
        return []

    # Filter for genuine signatures for this account
    matches = dataset_df[
        (dataset_df['image_path1'].str.contains(account_id, na=False)) & 
        (dataset_df['label'] == 1)
    ]

    if matches.empty:
        return []

    # Get unique paths
    ref_paths = matches['image_path1'].unique()[:num_refs]
    full_paths = []
    
    for ref_path in ref_paths:
        full_path = os.path.join(PROJECT_ROOT, bank, ref_path)
        if os.path.exists(full_path):
            full_paths.append(full_path)

    return full_paths


# -----------------------------
# Image Preprocessing
# -----------------------------

def preprocess_image(image_obj):

    img = np.array(image_obj.convert('L'))

    img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    return torch.from_numpy(img)


# -----------------------------
# Input Section
# -----------------------------

account_id = st.text_input("Enter Account ID", placeholder="e.g., ACC0001")

st.markdown("### Upload Signature")

uploaded_file = st.file_uploader(
    "Choose File",
    type=["png", "jpg", "jpeg", "tif", "bmp"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    img_display = Image.open(uploaded_file)

    st.image(img_display, caption="Uploaded Signature", use_column_width=True)


# -----------------------------
# Verification
# -----------------------------

if st.button("Verify Signature"):

    if not account_id:

        st.warning("Please enter an Account ID.")

    elif uploaded_file is None:

        st.warning("Please upload a signature image.")

    elif model is None:

        st.error(f"Model not found for {bank}. Please train the model first.")

    else:

        ref_paths = get_reference_signatures(account_id)

        if not ref_paths:
            st.error(f"No genuine reference signatures found for Account ID: {account_id}")
        else:
            with st.spinner("Analyzing signature against multiple references..."):
                upload_img_pil = Image.open(uploaded_file)
                upload_tensor = preprocess_image(upload_img_pil)
                
                distances = []
                
                with torch.no_grad():
                    # Compare with each reference signature
                    for ref_path in ref_paths:
                        ref_img_pil = Image.open(ref_path)
                        ref_tensor = preprocess_image(ref_img_pil)
                        
                        output1, output2 = model(ref_tensor, upload_tensor)
                        dist = nn.functional.pairwise_distance(output1, output2).item()
                        distances.append(dist)

                # Calculate mean distance (average of multiple comparisons)
                final_distance = np.mean(distances)
                
                # Verification Logic
                is_genuine = final_distance < threshold
                confidence = max(0, min(100, (1 - (final_distance / 2.0)) * 100))

                # Display Results
                st.markdown("---")
                if is_genuine:
                    st.success("### Status: Signature is Genuine ✅")
                else:
                    st.error("### Status: Signature is Forged! ❌")

                # Metrics Visualization
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Distance Score", f"{final_distance:.4f}")
                col_m2.metric("Threshold", f"{threshold:.2f}")
                col_m3.metric("Similarity Index", f"{confidence:.2f}%")

                st.progress(confidence / 100.0)

                st.markdown("---")
                st.markdown("### Reference Comparison")
                st.write(f"Analyzed against {len(ref_paths)} stored genuine signatures.")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(Image.open(ref_paths[0]), caption="Reference Signature (Primary)")
                with col2:
                    st.image(upload_img_pil, caption="Uploaded Signature")