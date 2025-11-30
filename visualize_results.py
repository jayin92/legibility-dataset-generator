import streamlit as st
import json
import os
from PIL import Image
import config

st.set_page_config(layout="wide", page_title="VLM Legibility Visualizer")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def main():
    st.title("VLM Legibility Visualizer")

    # Sidebar for file selection
    jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
    if not jsonl_files:
        st.error("No .jsonl files found in the current directory.")
        return

    selected_file = st.sidebar.selectbox("Select Result File", jsonl_files)
    
    # Load data
    data = load_jsonl(selected_file)
    if not data:
        st.warning("Selected file is empty or invalid.")
        return

    st.sidebar.write(f"Total Pairs: {len(data)}")

    # Pagination
    if 'index' not in st.session_state:
        st.session_state.index = 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous"):
            st.session_state.index = max(0, st.session_state.index - 1)
    with col2:
        st.slider("Go to index", 0, len(data) - 1, st.session_state.index, key='slider_index', on_change=lambda: st.session_state.update(index=st.session_state.slider_index))
    with col3:
        if st.button("Next"):
            st.session_state.index = min(len(data) - 1, st.session_state.index + 1)

    current_index = st.session_state.index
    row = data[current_index]

    # Display Content
    st.markdown(f"### Pair {current_index + 1} / {len(data)}")

    # Images
    col_a, col_b = st.columns(2)
    
    choice = row.get('choice', '').lower()
    
    # Determine highlight color
    color_a = "white"
    color_b = "white"
    
    if choice == "a_much_better":
        color_a = "green"
    elif choice == "a_better" or choice == "a":
        color_a = "lightgreen"
    elif choice == "b_much_better":
        color_b = "green"
    elif choice == "b_better" or choice == "b":
        color_b = "lightgreen"
    elif choice == "equal":
        color_a = "orange"
        color_b = "orange"

    with col_a:
        st.markdown(f"<h3 style='text-align: center; color: {color_a};'>Image A</h3>", unsafe_allow_html=True)
        try:
            image_a_path = os.path.join(config.OUTPUT_DIR, row['image_a'])
            image_a = Image.open(image_a_path)
            st.image(image_a, width="stretch") 
            st.caption(row['image_a'])
        except Exception as e:
            st.error(f"Error loading Image A: {e}")

    with col_b:
        st.markdown(f"<h3 style='text-align: center; color: {color_b};'>Image B</h3>", unsafe_allow_html=True)
        try:
            image_b_path = os.path.join(config.OUTPUT_DIR, row['image_b'])
            image_b = Image.open(image_b_path)
            st.image(image_b, width="stretch")
            st.caption(row['image_b'])
        except Exception as e:
            st.error(f"Error loading Image B: {e}")

    # Result
    st.divider()
    st.subheader("Model Decision")
    
    st.markdown(f"**Choice:** `{row.get('choice', 'N/A')}`")
    st.markdown(f"**Reasoning:**")
    st.info(row.get('reasoning', 'No reasoning provided.'))

    # Raw Data Expander
    with st.expander("View Raw JSON"):
        st.json(row)

if __name__ == "__main__":
    main()
