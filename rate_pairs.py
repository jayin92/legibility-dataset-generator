import os
import asyncio
import json
import pandas as pd
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import config # Assuming config.py exists and defines PAIR_CSV_FILE and OUTPUT_DIR
import re
import argparse
from create_composite import create_side_by_side_image

# --- Configuration ---
OUTPUT_CSV = "ratings.csv"
OUTPUT_DIR = config.OUTPUT_DIR
MODEL_NAME = "gemini-2.5-pro"
CONCURRENT_REQUESTS = 10 # Number of parallel API calls

# --- Prompts ---
def get_prompt_with_reasoning(letter, case_str):
    return (
        f"You are an expert in typography and legibility. Your task is to determine which of the two provided images of the {case_str} character, '{letter}', is more legible.\n\n"
        "First, provide a brief, step-by-step reasoning for your choice. Consider factors like clarity of form, distortion, ambiguity, and stroke consistency.\n\n"
        "Second, conclude with your final choice.\n\n"
        "Respond in a JSON format with two keys: \"reasoning\" and \"choice\". The \"choice\" value must be one of 'A', 'B', or 'equal'."
    )

def get_prompt_no_reasoning(letter, case_str):
    return (
        f"You are an expert in typography and legibility. Your task is to determine which of the two provided images of the {case_str} character, '{letter}', is more legible.\n\n"
        "Respond in a JSON format with one key: \"choice\". The \"choice\" value must be one of 'A', 'B', or 'equal'."
    )

async def rate_one_pair(model, row, semaphore, prompt_text, legacy_composite):
    """Makes a single API call to Gemini to rate a pair of images, with retries."""
    async with semaphore:
        image_a_path = os.path.join(OUTPUT_DIR, row.image_a)
        image_b_path = os.path.join(OUTPUT_DIR, row.image_b)
        
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"composite_{os.path.basename(image_a_path)}_{os.path.basename(image_b_path)}.png")
        composite_image = create_side_by_side_image(image_a_path, image_b_path, output_path=debug_path, legacy=legacy_composite)
        if composite_image is None:
            return {**row.to_dict(), 'choice': 'image_load_error', 'reasoning': ''}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await model.generate_content_async([prompt_text, composite_image])
                
                text_response = response.text.strip()
                json_match = re.search(r'```json\n(.*?)\n```', text_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text_response

                data = json.loads(json_str)
                choice = data.get('choice', '').lower()
                reasoning = data.get('reasoning', '')
                
                # If choice is valid, we are done.
                if choice in ['a', 'b', 'equal']:
                    return {**row.to_dict(), 'choice': choice, 'reasoning': reasoning}
                
                # If choice is invalid, loop will continue after a short delay.
                
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                # Malformed JSON or response, treat as a failed attempt
                if attempt == max_retries - 1:
                    return {**row.to_dict(), 'choice': 'json_error_after_retries', 'reasoning': str(response.text if 'response' in locals() else e)}
            except Exception as e:
                # API or other unexpected error
                if attempt == max_retries - 1:
                    return {**row.to_dict(), 'choice': 'api_error_after_retries', 'reasoning': str(e)}

            # Wait a bit before retrying if the loop hasn't exited
            await asyncio.sleep(1)
            
        # If all retries fail without a valid choice
        return {**row.to_dict(), 'choice': 'invalid_choice_after_retries', 'reasoning': 'Failed to get a valid choice after 5 attempts.'}

async def main():
    """Main function to read pairs, run concurrent API calls, and save results."""
    parser = argparse.ArgumentParser(description="Rate image pairs using the Gemini API.")
    parser.add_argument("input_csv", help="Path to the input CSV file containing image pairs.")
    parser.add_argument('--with-reasoning', action='store_true', help="Use the prompt with the reasoning step.")
    parser.add_argument('--legacy-composite', action='store_true', help="Use the legacy composite image creation logic.")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return
        
    df = pd.read_csv(args.input_csv)
    if df.empty:
        print("Input CSV is empty. Nothing to do.")
        return

    print(f"Found {len(df)} pairs to rate in {args.input_csv}.")

    if args.with_reasoning:
        print("--- Running in WITH REASONING mode ---")
    else:
        print("--- Running in NO REASONING mode ---")

    print("--- Gemini Pairwise Rating Script ---")
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Please create a .env file.")
        return
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []
    for _, row in df.iterrows():
        try:
            image_a_path = os.path.join(OUTPUT_DIR, row.image_a)
            dir_name = os.path.basename(os.path.dirname(image_a_path))
            parts = dir_name.split('_')
            letter = parts[0]
            case_info = parts[1] if len(parts) > 1 else ""
            case_str = f"{case_info}case" if case_info else ""
        except (IndexError, AttributeError):
            letter = "the character"
            case_str = ""

        prompt_text_func = get_prompt_with_reasoning if args.with_reasoning else get_prompt_no_reasoning
        prompt_text = prompt_text_func(letter, case_str)

        tasks.append(rate_one_pair(model, row, semaphore, prompt_text, args.legacy_composite))
    
    print(f"Starting rating process with {CONCURRENT_REQUESTS} concurrent requests...")
    results = await tqdm_asyncio.gather(*tasks)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n--- Rating Complete ---")
    print(f"Results saved to {OUTPUT_CSV}")
    
    rating_counts = results_df['choice'].value_counts()
    print("\nRating Summary:")
    print(rating_counts)

if __name__ == "__main__":
    os.makedirs("debug", exist_ok=True)
    asyncio.run(main())