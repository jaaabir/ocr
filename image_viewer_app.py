import streamlit as st
import os
import glob
from PIL import Image
from transformers import AutoImageProcessor
import cv2
import json 
import numpy as np 

def read_json(fname, encoding='utf-8'):
    data = []

    # Open the JSONL file and read each line
    with open(fname, 'r', encoding=encoding) as f:
        for line in f:
            try:
                # Parse each line as a JSON object
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(f"Error: {e}")

    json_metadata = {}
    for jdata in data:
        fname = jdata['file_name']
        gt = json.loads(jdata['ground_truth'])['gt_parse']['text_sequence']
        json_metadata[fname] = gt
    return json_metadata

st.set_page_config(page_title="Image Viewer", layout="wide")

st.title("📂 Image Viwer")

# --- Folder Browser ---
json_fname = 'metadata.jsonl'
encoder_model = "microsoft/dit-base-finetuned-rvlcdip"
image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)
folder_path = st.sidebar.text_input("Root Folder", value=".")
subfolders = [os.path.basename(f.path) for f in os.scandir(folder_path) if f.is_dir()] if os.path.exists(folder_path) else []
selected_folder = st.sidebar.selectbox("Select a subfolder", options=[""] + subfolders)

# --- Pagination Controls ---
n_per_page = st.sidebar.number_input("Images per page", min_value=1, max_value=100, value=8, step=1)

# --- Toggle: Show processed vs. original ---
show_processed = st.sidebar.radio("Show processed image", options=["Off", "On"], index=0)

# keep page state
if "page" not in st.session_state:
    st.session_state.page = 1

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Previous") and st.session_state.page > 1:
        st.session_state.page -= 1
with col2:
    st.session_state.page = st.number_input("Page", min_value=1, value=st.session_state.page, step=1)
with col3:
    if st.button("Next ➡️"):
        st.session_state.page += 1

# --- Load Images ---
sf = selected_folder
selected_folder = os.path.join(folder_path, selected_folder)
json_path = os.path.join(selected_folder, json_fname)
metadata = read_json(json_path) if os.path.exists(json_path) else {}

if selected_folder and os.path.exists(selected_folder):
    image_paths = sorted(glob.glob(os.path.join(selected_folder, "*.jpg")))

    if len(image_paths) == 0:
        st.warning("No images found in this folder.")
    else:
        total_pages = (len(image_paths) + n_per_page - 1) // n_per_page
        st.session_state.page = min(st.session_state.page, total_pages)

        start_idx = (st.session_state.page - 1) * n_per_page
        end_idx = min(start_idx + n_per_page, len(image_paths))

        st.write(f"Showing **{start_idx+1} - {end_idx}** of {len(image_paths)} images "
                 f"(Page {st.session_state.page}/{total_pages})")

        # --- Display Images ---
        cols = st.columns(4)  # adjust grid size
        for i, img_path in enumerate(image_paths[start_idx:end_idx]):
            with cols[i % 4]:
                try:
                    data = metadata[os.path.basename(img_path)]
                    if show_processed == "On":
                        t = selected_folder.replace('\\', '/') + '/' + os.path.basename(img_path)
                        img = cv2.cvtColor(cv2.imread(t), cv2.COLOR_BGR2RGB)
                        img = image_processor(img, return_tensors="pt").pixel_values.squeeze(0).permute(1, -1, 0).numpy().clip(0, 1)
                        st.image(img, caption=f"[Processed] {os.path.basename(img_path)} \n\n {data}", use_container_width=True)
                    else:
                        img = Image.open(img_path)
                        img = img.resize((224, 224), Image.Resampling.BICUBIC)
                        st.image(img, caption=f"{os.path.basename(img_path)} \n\n {data}", use_container_width=True)

                except Exception as e:
                    st.error(f"Error loading {img_path}: {e}")
else:
    st.info("👆 Select a folder from the sidebar.")
