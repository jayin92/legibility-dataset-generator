import os
# Set Gradio temp dir to local to avoid permission issues in /tmp
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_temp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import argparse
import torch
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor

# Import model components from training script
from train_legibility_predictor import (
    LegibilityPredictor,
    SigLIPFeatureExtractor,
    RankNetScorer
)

# Fallback if SIGLIP_MODEL_NAME was removed from global scope in previous edits
try:
    from train_legibility_predictor import SIGLIP_MODEL_NAME
except ImportError:
    SIGLIP_MODEL_NAME = "google/siglip2-so400m-patch16-512"

def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    
    # Initialize model components
    feature_extractor = SigLIPFeatureExtractor(SIGLIP_MODEL_NAME)
    input_dim = feature_extractor.get_output_dim()
    
    # We need to know hidden_dim from training, defaulting to 256 as per script default
    # Ideally this should be saved in the checkpoint or config
    scorer = RankNetScorer(input_dim=input_dim, hidden_dim=256) 
    
    model = LegibilityPredictor(feature_extractor, scorer)
    
    # Load state dict
    # The training script saves the *whole* model state dict in checkpoint_latest/best
    # or just the scorer in scorer_final.pth. We need to handle both.
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle torch.compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Full model load failed: {e}")
        print("Trying to load as scorer only...")
        # If it's just the scorer, the keys might still have prefixes or be different
        # Let's try loading into model.scorer if keys match
        try:
            model.scorer.load_state_dict(state_dict)
        except RuntimeError as e2:
             print(f"Scorer load failed: {e2}")
             raise e2
            
    model.to(device)
    model.eval()
    return model

def predict(image, model, processor, device):
    if image is None:
        return 0.0
    
    # Convert to PIL if needed (Gradio might return numpy array)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    
    # Preprocess
    # SigLIP expects specific input size, processor handles resizing usually
    # But for drawing canvas (often white on black or vice versa), we might need to invert?
    # The dataset generator renders black text on white background.
    # Gradio sketchpad usually gives black on white or transparent.
    # Let's ensure it's RGB.
    
    # Check if image is empty/blank
    if image.getbbox() is None:
        return 0.0

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        score = model.score(pixel_values)
        
    return float(score.item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

    # Define Gradio interface
    def predict_fn(sketch):
        # Sketchpad returns a dict with 'composite', 'layers', 'background'
        if isinstance(sketch, dict):
            image = sketch.get("composite")
        else:
            image = sketch
            
        return predict(image, model, processor, device)

    with gr.Blocks() as demo:
        gr.Markdown("# Legibility Predictor Demo")
        gr.Markdown("Draw a character below to check its legibility score.")
        
        with gr.Row():
            with gr.Column():
                # Use Sketchpad for drawing
                canvas = gr.Sketchpad(
                    label="Draw Character", 
                    type="numpy",
                    brush=gr.Brush(colors=["#000000"], color_mode="fixed"),
                    canvas_size=(512, 512)
                )
                submit_btn = gr.Button("Predict Legibility")
            
            with gr.Column():
                score_output = gr.Number(label="Legibility Score (0-1)")
                
        submit_btn.click(
            fn=predict_fn,
            inputs=[canvas],
            outputs=[score_output]
        )

    demo.launch()

if __name__ == "__main__":
    main()
