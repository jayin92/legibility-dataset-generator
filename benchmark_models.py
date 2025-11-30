import os
import asyncio
import json
import argparse
import re
import time
import random
import base64
from collections import Counter
import google.generativeai as genai
import openai
from PIL import Image
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
MODELS_TO_TEST = {
    "gemini": [
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        # "gemini-2.5-flash-preview-09-2025",
        # "gemini-2.5-flash-lite-preview-09-2025"
    ],
    "qwen": [
        # "qwen-vl-plus",
        "qwen-vl-ocr"
    ],
    "openai": [
        "gpt-5-mini",
        "gpt-5"
    ]
}
CONCURRENT_REQUESTS = 5

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
        "Respond with a single word: 'A', 'B', or 'equal'."
    )

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

async def get_gemini_rating(session, images, semaphore, prompt_text, mode, five_level=False):
    """Makes a single API call to the Gemini API with retry logic."""
    max_retries = 5
    base_delay = 2
    valid_choices = ['a_much_better', 'a_better', 'equal', 'b_better', 'b_much_better'] if five_level else ['a', 'b', 'equal']
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                if isinstance(images, list) and len(images) == 2:
                    # Interleave images with labels for clarity
                    content = ["Image A:", images[0], "Image B:", images[1], prompt_text]
                else:
                    # Single image case (or unexpected list)
                    image = images[0] if isinstance(images, list) else images
                    content = [prompt_text, image]

                response = await session.generate_content_async(content)
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
                    # Basic cleanup for single word response
                    choice = choice.strip().strip('"').strip("'")
                    if choice not in valid_choices:
                         # Try to find the choice in the text if it's verbose
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
                        print(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                return {"choice": "error", "reasoning": str(e)}
        
        return {"choice": "error", "reasoning": "Max retries exceeded"}

async def get_openai_rating(client, provider, model_name, images, semaphore, prompt_text, mode, five_level=False):
    """Makes a single API call to an OpenAI-compatible API."""
    valid_choices = ['a_much_better', 'a_better', 'equal', 'b_better', 'b_much_better'] if five_level else ['a', 'b', 'equal']
    async with semaphore:
        try:
            extra_kwargs = {}
            if provider == "qwen" and model_name != "qwen-vl-plus":
                extra_kwargs['extra_body'] = {"enable_thinking": True, "thinking_budget": 4000}

            messages_content = []
            
            if isinstance(images, list) and len(images) == 2:
                # Multi-image case
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
            else:
                # Single image case
                image = images[0] if isinstance(images, list) else images
                base64_image = image_to_base64(image)
                messages_content = [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
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

async def benchmark_one_model(provider, model_name, images, num_runs, prompt_text, mode_name, api_keys, five_level=False):
    """Runs a full benchmark for a single model and a single prompt mode."""
    print(f"\n--- Benchmarking Model: {model_name} ({mode_name}) ---")
    
    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    if provider == "gemini":
        try:
            model = genai.GenerativeModel(model_name)
            tasks = [get_gemini_rating(model, images, semaphore, prompt_text, mode_name, five_level) for _ in range(num_runs)]
        except Exception as e:
            print(f"Error initializing model: {e}")
            return {"model_name": model_name, "mode_name": mode_name, "error": "Initialization failed"}
    
    elif provider == "openai" or provider == "qwen":
        api_key_name = "openai" if provider == "openai" else "dashscope"
        api_key = api_keys.get(api_key_name)
        base_url = "https://api.openai.com/v1" if provider == "openai" else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        if not api_key:
            return {"model_name": model_name, "mode_name": mode_name, "error": f"{api_key_name.upper()}_API_KEY not found"}
        
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        tasks = [get_openai_rating(client, provider, model_name, images, semaphore, prompt_text, mode_name, five_level) for _ in range(num_runs)]

    else:
        return {"model_name": model_name, "mode_name": mode_name, "error": "Unknown provider"}

    results = await tqdm_asyncio.gather(*tasks, desc=f"Querying {model_name} ({mode_name})")
    
    # Analyze results
    choices = [r.get("choice", "error") for r in results]
    counts = Counter(choices)
    
    # Calculate consistency (percentage of the most frequent choice)
    if choices:
        most_common = counts.most_common(1)
        if most_common:
            consistency = (most_common[0][1] / len(choices)) * 100
            top_choice = most_common[0][0]
        else:
            consistency = 0
            top_choice = "N/A"
    else:
        consistency = 0
        top_choice = "N/A"

    # Calculate speed (approximate, based on total time would be better but this is per-request avg)
    # Since we run concurrently, total time is better.
    # For now, we don't have total time easily without modifying the loop structure significantly.
    # We'll just report the distribution.
    
    return {
        "model_name": model_name,
        "mode_name": mode_name,
        "consistency": consistency,
        "top_choice": top_choice,
        "distribution": dict(counts),
        "raw_results": results
    }

async def main(image_paths, num_runs, five_level_scores):
    """Main function to loop through models and prompt modes for benchmarking."""
    print(f"--- Model Ablation Study ---")
    print(f"Images: {image_paths}")
    print(f"Runs per condition: {num_runs}")
    print(f"Five-level scoring: {five_level_scores}\n")

    load_dotenv()
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "dashscope": os.getenv("DASHSCOPE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }

    if not api_keys["gemini"]:
        print("Warning: GEMINI_API_KEY not found. Skipping Gemini models.")
    
    genai.configure(api_key=api_keys["gemini"])

    images = []
    for p in image_paths:
        try:
            images.append(Image.open(p))
        except FileNotFoundError:
            print(f"Error: Image file not found at {p}")
            return

    # Extract letter and case from first image path
    first_image_path = image_paths[0]
    try:
        dir_name = os.path.basename(os.path.dirname(first_image_path))
        parts = dir_name.split('_')
        letter = parts[0]
        case_info = parts[1] if len(parts) > 1 else ""
        case_str = f"{case_info}case" if case_info else ""
        print(f"Letter: {letter}, Case: {case_str}" )
    except IndexError:
        print("Warning: Could not determine letter from image path. Using generic prompt.")
        letter = "the character"
        case_str = ""

    # Determine mode and prompts based on number of images
    models_to_run = MODELS_TO_TEST.copy()
    
    if len(images) == 2:
        print("Two images provided. Running in separate-image mode.")
        if five_level_scores:
             prompt_modes = {
                "With_Reasoning": get_prompt_with_reasoning_five_level(letter, case_str),
                "No_Reasoning": get_prompt_no_reasoning_five_level(letter, case_str)
            }
        else:
            prompt_modes = {
                "With_Reasoning": get_prompt_with_reasoning_two_images(letter, case_str),
                "No_Reasoning": get_prompt_no_reasoning_two_images(letter, case_str)
            }
    elif len(images) == 1:
        print("Single image provided. Running in composite-image mode.")
        if five_level_scores:
             print("Warning: Five-level scoring not implemented for single composite image mode. Using standard prompts.")
        prompt_modes = {
            "With_Reasoning": get_prompt_with_reasoning(letter, case_str),
            "No_Reasoning": get_prompt_no_reasoning(letter, case_str)
        }
    else:
        print("Error: Please provide either 1 composite image or 2 separate images.")
        return

    benchmark_results = []
    start_time = time.time()

    for provider, models in models_to_run.items():
        for model_name in models:
            for mode_name, prompt in prompt_modes.items():
                mode_start_time = time.time()
                result = await benchmark_one_model(provider, model_name, images, num_runs, prompt, mode_name, api_keys, five_level_scores)
                mode_duration = time.time() - mode_start_time
                
                # Calculate speed (req/s)
                speed = num_runs / mode_duration if mode_duration > 0 else 0
                result["speed"] = speed
                
                benchmark_results.append(result)
                
                if "raw_results" in result and result["raw_results"]:
                    output_filename = f"benchmark_results_{model_name}_{mode_name}.json"
                    with open(output_filename, 'w') as f:
                        json.dump(result["raw_results"], f, indent=2)
                    print(f"Saved detailed results to {output_filename}")

    # --- Final Report ---
    print("\n--- Ablation Study Summary ---")
    header = "{:<45} | {:<18} | {:<15} | {:<15} | {:<12} | {}".format('Model', 'Prompt Mode', 'Speed (req/s)', 'Consistency', 'Top Choice', 'Distribution')
    print(header)
    print("-" * len(header))

    for res in benchmark_results:
        if "error" in res:
             print("{:<45} | {:<18} | {:<15} | {:<15} | {:<12} | {}".format(
                res['model_name'], res['mode_name'], "N/A", "N/A", "Error", res['error']
            ))
             continue
             
        dist_str = ", ".join([f"{k}: {v}" for k, v in res['distribution'].items()])
        print("{:<45} | {:<18} | {:<15.2f} | {:<15} | {:<12} | {}".format(
            res['model_name'], 
            res['mode_name'], 
            res['speed'],
            f"{res['consistency']:.1f}%",
            res['top_choice'],
            dist_str
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study for Gemini models.")
    parser.add_argument("image_paths", nargs='+', help="Path(s) to the image file(s). Provide 1 for composite, 2 for separate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per condition.")
    parser.add_argument("--five-level-scores", action='store_true', help="Enable five-level scoring (A>>B, A>B, Equal, B>A, B>>A).")
    args = parser.parse_args()
    
    asyncio.run(main(args.image_paths, args.runs, args.five_level_scores))