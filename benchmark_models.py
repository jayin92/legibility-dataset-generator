import os
import asyncio
import json
import argparse
import re
import time
import base64
from collections import Counter
import google.generativeai as genai
import openai
from PIL import Image
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# --- Configuration ---
MODELS_TO_TEST = {
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        # "gemini-2.5-flash-preview-09-2025",
        # "gemini-2.5-flash-lite-preview-09-2025"
    ],
    "qwen": [
        "qwen-vl-plus",
        "qwen-vl-ocr"
    ],
    "openai": [
        "gpt-5-mini",
        "gpt-5"
    ]
}
CONCURRENT_REQUESTS = 10

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

def image_to_base64(image):
    """Converts a PIL image to a base64 encoded string."""
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def get_gemini_rating(session, image, semaphore, prompt_text, mode):
    """Makes a single API call to the Gemini API."""
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

async def get_openai_rating(client, provider, model_name, image, semaphore, prompt_text, mode):
    """Makes a single API call to an OpenAI-compatible API."""
    async with semaphore:
        try:
            base64_image = image_to_base64(image)
            
            extra_kwargs = {}
            if provider == "qwen":
                extra_kwargs['extra_body'] = {"enable_thinking": True, "thinking_budget": 4000}

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
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
                if choice not in ['a', 'b', 'equal']:
                    choice = 'invalid_choice'
                return {"choice": choice, "reasoning": ""}

        except Exception as e:
            return {"choice": "error", "reasoning": str(e)}

async def benchmark_one_model(provider, model_name, image, num_runs, prompt_text, mode_name, api_keys):
    """Runs a full benchmark for a single model and a single prompt mode."""
    print(f"\n--- Benchmarking Model: {model_name} ({mode_name}) ---")
    
    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    if provider == "gemini":
        try:
            model = genai.GenerativeModel(model_name)
            tasks = [get_gemini_rating(model, image, semaphore, prompt_text, mode_name) for _ in range(num_runs)]
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
        tasks = [get_openai_rating(client, provider, model_name, image, semaphore, prompt_text, mode_name) for _ in range(num_runs)]

    else:
        return {"model_name": model_name, "mode_name": mode_name, "error": "Unknown provider"}

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
    print(f"--- Model Ablation Study ---")
    print(f"Image: {image_path}")
    print(f"Runs per condition: {num_runs}\n")

    load_dotenv()
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "dashscope": os.getenv("DASHSCOPE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }

    if not api_keys["gemini"]:
        print("Warning: GEMINI_API_KEY not found. Skipping Gemini models.")

    genai.configure(api_key=api_keys["gemini"])

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    # Extract letter and case from image path
    try:
        dir_name = os.path.basename(os.path.dirname(image_path))
        parts = dir_name.split('_')
        letter = parts[0]
        case_info = parts[1] if len(parts) > 1 else ""
        case_str = f"{case_info}case" if case_info else ""
    except IndexError:
        print("Warning: Could not determine letter from image path. Using generic prompt.")
        letter = "the character"
        case_str = ""

    prompt_modes = {
        "With_Reasoning": get_prompt_with_reasoning(letter, case_str),
        "No_Reasoning": get_prompt_no_reasoning(letter, case_str)
    }

    benchmark_results = []
    for provider, models in MODELS_TO_TEST.items():
        for model_name in models:
            for mode_name, prompt in prompt_modes.items():
                result = await benchmark_one_model(provider, model_name, image, num_runs, prompt, mode_name, api_keys)
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
    parser = argparse.ArgumentParser(description="Benchmark various models for speed and consistency.")
    parser.add_argument("image_path", help="Path to the composite image file to test.")
    parser.add_argument("--runs", type=int, default=20, help="Number of times to query the API for each model.")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.image_path, args.runs))