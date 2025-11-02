import os
import asyncio
import json
import argparse
import re
import time
from collections import Counter
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# --- Configuration ---
MODELS_TO_TEST = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview-09-2025"
]
CONCURRENT_REQUESTS = 10

# --- Prompts ---
PROMPT_WITH_REASONING = (
    "You are an expert in typography and legibility. Your task is to determine which of the two provided images of the character, 'A' or 'B', is more legible.\n\n"
    "First, provide a brief, step-by-step reasoning for your choice. Consider factors like clarity of form, distortion, ambiguity, and stroke consistency.\n\n"
    "Second, conclude with your final choice.\n\n"
    "Respond in a JSON format with two keys: \"reasoning\" and \"choice\". The \"choice\" value must be one of 'A', 'B', or 'equal'."
)

PROMPT_NO_REASONING = (
    "You are an expert in typography and legibility. Your task is to determine which of the two provided images of the character, 'A' or 'B', is more legible.\n\n"
    "Respond with a single word: 'A', 'B', or 'equal'."
)


async def get_one_rating(session, image, semaphore, prompt_text, mode):
    """Makes a single API call to get a rating for one image, handling different modes."""
    async with semaphore:
        try:
            response = await session.generate_content_async([prompt_text, image])
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
                if choice not in ['a', 'b', 'equal']:
                    choice = 'invalid_choice'
                return {"choice": choice, "reasoning": ""}

        except Exception as e:
            return {"choice": "error", "reasoning": str(e)}

async def benchmark_one_model(model_name, image, num_runs, prompt_text, mode_name):
    """Runs a full benchmark for a single model and a single prompt mode."""
    print(f"\n--- Benchmarking Model: {model_name} ({mode_name}) ---")
    
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return {"model_name": model_name, "mode_name": mode_name, "error": "Initialization failed"}

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [get_one_rating(model, image, semaphore, prompt_text, mode_name) for _ in range(num_runs)]
    
    start_time = time.time()
    results = await tqdm_asyncio.gather(*tasks, desc=f"Querying {model_name} ({mode_name})")
    end_time = time.time()
    
    total_time = end_time - start_time
    reqs_per_sec = num_runs / total_time if total_time > 0 else float('inf')
    
    choices = [r['choice'] for r in results]
    choice_counts = Counter(choices)
    
    consistency = 0.0
    most_common_choice = "N/A"
    if choice_counts:
        most_common_choice, most_common_count = choice_counts.most_common(1)[0]
        if most_common_choice not in ['error', 'parse_error', 'invalid_choice']:
            consistency = (most_common_count / num_runs) * 100

    return {
        "model_name": model_name,
        "mode_name": mode_name,
        "total_time": total_time,
        "reqs_per_sec": reqs_per_sec,
        "consistency": consistency,
        "most_common_choice": most_common_choice,
        "choice_counts": choice_counts,
        "raw_results": results
    }

async def main(image_path, num_runs):
    """Main function to loop through models and prompt modes for benchmarking."""
    print(f"--- Gemini Model Ablation Study ---")
    print(f"Image: {image_path}")
    print(f"Runs per condition: {num_runs}\n")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Please create a .env file.")
        return
    genai.configure(api_key=api_key)

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    prompt_modes = {
        "With_Reasoning": PROMPT_WITH_REASONING,
        "No_Reasoning": PROMPT_NO_REASONING
    }

    benchmark_results = []
    for model_name in MODELS_TO_TEST:
        for mode_name, prompt in prompt_modes.items():
            result = await benchmark_one_model(model_name, image, num_runs, prompt, mode_name)
            benchmark_results.append(result)

            if "raw_results" in result:
                output_filename = f"benchmark_results_{model_name}_{mode_name}.json"
                with open(output_filename, 'w') as f:
                    json.dump(result["raw_results"], f, indent=2)
                print(f"Saved detailed results to {output_filename}")

    # --- Final Report ---
    print("\n--- Ablation Study Summary ---")
    header = f"{'Model':<45} | {'Prompt Mode':<18} | {'Speed (req/s)':<15} | {'Consistency':<15} | {'Top Choice':<12} | {'Distribution'}"
    print(header)
    print("-" * len(header))
    
    for result in benchmark_results:
        if result.get("error"):
            print(f"{result['model_name']:<45} | {result.get('mode_name', 'N/A'):<18} | {result['error']:<15}")
            continue
            
        model_name = result['model_name']
        mode_name = result['mode_name']
        speed = f"{result['reqs_per_sec']:.2f}"
        consistency = f"{result['consistency']:.1f}%"
        top_choice = result['most_common_choice']
        distribution = ", ".join([f"{k}: {v}" for k, v in result['choice_counts'].items()])
        
        print(f"{model_name:<45} | {mode_name:<18} | {speed:<15} | {consistency:<15} | {top_choice:<12} | {distribution}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gemini models for speed and consistency.")
    parser.add_argument("image_path", help="Path to the composite image file to test.")
    parser.add_argument("--runs", type=int, default=20, help="Number of times to query the API for each model.")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.image_path, args.runs))