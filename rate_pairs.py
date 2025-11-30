import os
import asyncio
import json
import pandas as pd
import google.generativeai as genai
import openai
from PIL import Image
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import config
import re
import argparse
import random
import base64

# --- Configuration ---
OUTPUT_CSV = "ratings.csv"
OUTPUT_DIR = config.OUTPUT_DIR
CONCURRENT_REQUESTS = 5

# --- Prompts ---
def get_prompt_with_reasoning_two_images(letter, case_str):
    return (
        f"You are an expert in typography and legibility. You have been provided with two separate images, labeled Image A and Image B. Your task is to determine which of these two images of the {case_str} character, '{letter}', is more legible.\n\n"
        "First, provide a brief, step-by-step reasoning for your choice. Consider factors like clarity of form, distortion, ambiguity, and stroke consistency.\n\n"
        "Second, conclude with your final choice.\n\n"
        "Respond in a JSON format with two keys: \"reasoning\" and \"choice\". The \"choice\" value must be one of 'A', 'B', or 'equal'."
    )

def get_prompt_no_reasoning_two_images(letter, case_str):
    return (
        f"You are an expert in typography and legibility. You have been provided with two separate images, labeled Image A and Image B. Your task is to determine which of these two images of the {case_str} character, '{letter}', is more legible.\n\n"
        "Respond with a single word: 'A', 'B', or 'equal'."
    )

def get_prompt_with_reasoning_five_level(letter, case_str):
    return (
        f"You are an expert in typography and legibility. You have been provided with two separate images, labeled Image A and Image B. Your task is to determine which of these two images of the {case_str} character, '{letter}', is more legible.\n\n"
        "First, provide a brief, step-by-step reasoning for your choice. Consider factors like clarity of form, distortion, ambiguity, and stroke consistency.\n\n"
        "Second, conclude with your final choice using one of the following 5 options:\n"
        "- 'a_much_better': Image A is significantly more legible than Image B.\n"
        "- 'a_better': Image A is somewhat more legible than Image B.\n"
        "- 'equal': Both images have similar legibility.\n"
        "- 'b_better': Image B is somewhat more legible than Image A.\n"
        "- 'b_much_better': Image B is significantly more legible than Image A.\n\n"
        "Respond in a JSON format with two keys: \"reasoning\" and \"choice\"."
    )

def get_prompt_no_reasoning_five_level(letter, case_str):
    return (
        f"You are an expert in typography and legibility. You have been provided with two separate images, labeled Image A and Image B. Your task is to determine which of these two images of the {case_str} character, '{letter}', is more legible.\n\n"
        "Respond with a single word representing your choice from the following options:\n"
        "- 'a_much_better'\n"
        "- 'a_better'\n"
        "- 'equal'\n"
        "- 'b_better'\n"
        "- 'b_much_better'"
    )

def image_to_base64(image):
    """Converts a PIL image to a base64 encoded string."""
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def get_gemini_rating(model, images, semaphore, prompt_text, mode, five_level=False):
    """Makes a single API call to the Gemini API with retry logic."""
    max_retries = 5
    base_delay = 2
    valid_choices = ['a_much_better', 'a_better', 'equal', 'b_better', 'b_much_better'] if five_level else ['a', 'b', 'equal']
    
    # Semaphore is now handled by the caller (rate_one_pair)
    for attempt in range(max_retries):
        try:
            # Interleave images with labels for clarity
            content = ["Image A:", images[0], "Image B:", images[1], prompt_text]

            response = await model.generate_content_async(content)
            text_response = response.text.strip()

            if mode == "With_Reasoning":
                json_match = re.search(r'```json\n(.*?)\n```', text_response, re.DOTALL)
                json_str = json_match.group(1) if json_match else text_response
                data = json.loads(json_str)
                choice = data.get('choice', 'parse_error').lower()
                reasoning = data.get('reasoning', 'parse_error')
                return {"choice": choice, "reasoning": reasoning}
            else: # No_Reasoning mode
                choice = text_response.lower()
                choice = choice.strip().strip('"').strip("'")
                if choice not in valid_choices:
                        for vc in valid_choices:
                            if vc in choice:
                                choice = vc
                                break
                if choice not in valid_choices:
                    choice = 'invalid_choice'
                return {"choice": choice, "reasoning": ""}

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    # print(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    continue
            return {"choice": "error", "reasoning": str(e)}
    
    return {"choice": "error", "reasoning": "Max retries exceeded"}

async def get_openai_rating(client, provider, model_name, images, semaphore, prompt_text, mode, five_level=False):
    """Makes a single API call to an OpenAI-compatible API."""
    valid_choices = ['a_much_better', 'a_better', 'equal', 'b_better', 'b_much_better'] if five_level else ['a', 'b', 'equal']
    
    # Semaphore is now handled by the caller (rate_one_pair)
    try:
        extra_kwargs = {}
        if provider == "qwen" and model_name != "qwen-vl-plus":
            extra_kwargs['extra_body'] = {"enable_thinking": True, "thinking_budget": 4000}

        base64_img_a = image_to_base64(images[0])
        base64_img_b = image_to_base64(images[1])
        
        messages_content = [
            {"type": "text", "text": "Image A:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img_a}"}
            },
            {"type": "text", "text": "Image B:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img_b}"}
            },
            {"type": "text", "text": prompt_text}
        ]

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": messages_content,
                }
            ],
            **extra_kwargs
        )
        text_response = response.choices[0].message.content.strip()

        if mode == "With_Reasoning":
            if not text_response:
                return {"choice": "error", "reasoning": "Empty response from model."}

            json_match = re.search(r'```json\n(.*?)\n```', text_response, re.DOTALL)
            json_str = json_match.group(1) if json_match else text_response
            
            try:
                data = json.loads(json_str)
                choice = data.get('choice', 'parse_error').lower()
                reasoning = data.get('reasoning', 'parse_error')
                return {"choice": choice, "reasoning": reasoning}
            except json.JSONDecodeError as e:
                return {"choice": "error", "reasoning": f"JSON parsing error: {e}. Raw response: {text_response}"}
        else: # No_Reasoning mode
            choice = text_response.lower()
            choice = choice.strip().strip('"').strip("'")
            if choice not in valid_choices:
                    for vc in valid_choices:
                        if vc in choice:
                            choice = vc
                            break
            if choice not in valid_choices:
                choice = 'invalid_choice'
            return {"choice": choice, "reasoning": ""}

    except Exception as e:
        return {"choice": "error", "reasoning": str(e)}

async def rate_one_pair(provider, model_name, client_or_model, row, semaphore, prompt_text, mode, file_lock, output_jsonl, five_level=False):
    """Rates a single pair of images."""
    image_a_path = os.path.join(OUTPUT_DIR, row.image_a)
    image_b_path = os.path.join(OUTPUT_DIR, row.image_b)
    
    async with semaphore:
        try:
            # Open images inside the semaphore block to avoid "Too many open files"
            image_a = Image.open(image_a_path)
            image_b = Image.open(image_b_path)
            images = [image_a, image_b]
            
            if provider == "gemini":
                result = await get_gemini_rating(client_or_model, images, semaphore, prompt_text, mode, five_level)
            else:
                result = await get_openai_rating(client_or_model, provider, model_name, images, semaphore, prompt_text, mode, five_level)
            
            # Close images explicitly to be safe
            image_a.close()
            image_b.close()
            
            output_data = {**row.to_dict(), 'choice': result['choice'], 'reasoning': result['reasoning']}
            
            # Write to JSONL file on-the-fly
            async with file_lock:
                with open(output_jsonl, "a") as f:
                    f.write(json.dumps(output_data) + "\n")
            
            return output_data
            
        except Exception as e:
            error_data = {**row.to_dict(), 'choice': 'image_load_error', 'reasoning': str(e)}
            async with file_lock:
                with open(output_jsonl, "a") as f:
                    f.write(json.dumps(error_data) + "\n")
            return error_data

async def main():
    parser = argparse.ArgumentParser(description="Rate image pairs using VLM APIs.")
    parser.add_argument("input_csv", help="Path to the input CSV file containing image pairs.")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai", "qwen"], help="Model provider.")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name.")
    parser.add_argument('--with-reasoning', action='store_true', help="Use the prompt with the reasoning step.")
    parser.add_argument("--five-level-scores", action='store_true', help="Enable five-level scoring (A>>B, A>B, Equal, B>A, B>>A).")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return
        
    df = pd.read_csv(args.input_csv)
    if df.empty:
        print("Input CSV is empty. Nothing to do.")
        return

    print(f"Found {len(df)} pairs to rate in {args.input_csv}.")
    mode = "With_Reasoning" if args.with_reasoning else "No_Reasoning"
    print(f"--- Running in {mode} mode with {args.provider} : {args.model} ---")
    print(f"Five-level scoring: {args.five_level_scores}")

    # Setup output JSONL file
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_jsonl = f"ratings_{base_name}_{args.provider}_{args.model}.jsonl"
    # Clear existing file if it exists
    with open(output_jsonl, "w") as f:
        pass
    print(f"Logging results on-the-fly to: {output_jsonl}")

    load_dotenv()
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "dashscope": os.getenv("DASHSCOPE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }

    client_or_model = None
    if args.provider == "gemini":
        if not api_keys["gemini"]:
            print("Error: GEMINI_API_KEY not found.")
            return
        genai.configure(api_key=api_keys["gemini"])
        client_or_model = genai.GenerativeModel(args.model)
    elif args.provider in ["openai", "qwen"]:
        api_key_name = "openai" if args.provider == "openai" else "dashscope"
        api_key = api_keys.get(api_key_name)
        if not api_key:
             print(f"Error: {api_key_name.upper()}_API_KEY not found.")
             return
        base_url = "https://api.openai.com/v1" if args.provider == "openai" else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        client_or_model = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()
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

        if args.five_level_scores:
            prompt_func = get_prompt_with_reasoning_five_level if args.with_reasoning else get_prompt_no_reasoning_five_level
        else:
            prompt_func = get_prompt_with_reasoning_two_images if args.with_reasoning else get_prompt_no_reasoning_two_images
        
        prompt_text = prompt_func(letter, case_str)

        tasks.append(rate_one_pair(args.provider, args.model, client_or_model, row, semaphore, prompt_text, mode, file_lock, output_jsonl, args.five_level_scores))
    
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
    asyncio.run(main())