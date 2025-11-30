import os
import argparse
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_calligram(word, shape, output_dir, model_name, api_key=None):
    """
    Generates a legible compact calligram using the Gemini API.
    """
    if api_key:
        genai.configure(api_key=api_key)
    elif os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        print("Error: No API key provided. Set GOOGLE_API_KEY in .env or pass --api_key.")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating calligram for word '{word}' in shape '{shape}' using model '{model_name}'...")

    # --- PROMPT ENGINEERING FOR NANO BANANA ---
    # Key elements:
    # 1. "Legible compact calligram": Defines the specific art style.
    # 2. "Vector graphic", "Silhouette": Enforces a clean, shape-based look.
    # 3. "Distorted typography": Explains how the text should behave.
    # 4. "White background", "Black text": standardizes output for dataset usage.
    
    prompt = f"""
    Generate a high-contrast, black-and-white vector graphic.
    
    Subject: A legible compact calligram of the word "{word}".
    Shape: The word must be distorted and arranged to perfectly fill the silhouette of a {shape}.
    
    Requirements:
    1. The entire shape of the {shape} must be formed SOLELY by the letters of the word "{word}".
    2. The letters must be distorted to fit the contours of the {shape} tightly, with minimal gaps.
    3. CRITICAL: The word "{word}" must remain LEGIBLE. Do not over-distort to the point of unreadability.
    4. Style: Flat vector art, solid black text on a pure white background. No shading, no gradients, no extra decorative elements.
    5. The letters should be bold and thick to define the shape clearly.
    """

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Check if the response contains an image
        if response.parts and hasattr(response.parts[0], 'image'):
            img = response.parts[0].image
            
            # Save the image
            filename = f"{word}_{shape.replace(' ', '_')}.png"
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
            print(f"Success! Image saved to: {output_path}")
            return output_path
        else:
            # Handle cases where the model refuses or returns text
            print("Generation failed or returned no image.")
            print("Feedback:", response.text if hasattr(response, 'text') else "No text feedback")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate Legible Compact Calligrams with Nano Banana (Gemini 3 Pro Image)")
    parser.add_argument("--word", type=str, required=True, help="The word to use in the calligram")
    parser.add_argument("--shape", type=str, required=True, help="The shape to fill (e.g., 'cat', 'heart')")
    parser.add_argument("--output_dir", type=str, default="outputs/calligrams", help="Directory to save outputs")
    # Defaulting to a likely model name for "Nano Banana Pro" / Gemini 3 Pro Image
    # User can override this if the actual string is different.
    parser.add_argument("--model_name", type=str, default="gemini-2.0-pro-exp-02-05", help="The Gemini model name to use")
    parser.add_argument("--api_key", type=str, help="Google Cloud API Key (optional if GOOGLE_API_KEY env var is set)")

    args = parser.parse_args()

    generate_calligram(args.word, args.shape, args.output_dir, args.model_name, args.api_key)

if __name__ == "__main__":
    main()
