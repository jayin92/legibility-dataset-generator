import os
import random
import csv
import argparse
import config
from tqdm import tqdm

def main():
    """Generates a CSV file with pairs of images for a specified character."""
    parser = argparse.ArgumentParser(description="VLM Pairwise Comparison List Generator.")
    parser.add_argument("--letter", required=True, help="The character to generate pairs for (e.g., 'A', 'b').")
    args = parser.parse_args()

    letter_to_process = args.letter
    print(f"--- VLM Pairwise Comparison List Generator for character: '{letter_to_process}' ---")
    
    if not os.path.exists(config.OUTPUT_DIR):
        print(f"Error: Output directory not found: {config.OUTPUT_DIR}")
        print("Please run 'generate_dataset.py' first.")
        return

    # 1. Scan output directory for the specified character
    print(f"Scanning for images of '{letter_to_process}'...")
    
    char_for_path = letter_to_process
    if 'a' <= letter_to_process <= 'z':
        dir_name = f"{char_for_path}_lower"
    elif 'A' <= letter_to_process <= 'Z':
        char_for_path = letter_to_process.lower()
        dir_name = f"{char_for_path}_upper"
    else:
        dir_name = char_for_path

    char_dir = os.path.join(config.OUTPUT_DIR, dir_name)
    if not os.path.exists(char_dir):
        print(f"Error: No images found for character '{letter_to_process}' in {char_dir}")
        return
        
    images = [os.path.join(dir_name, img_name) for img_name in os.listdir(char_dir) if img_name.endswith('.png')]
    
    if len(images) < 2:
        print(f"Error: Not enough images found for character '{letter_to_process}' to create pairs. Need at least 2, found {len(images)}.")
        return
        
    print(f"Found {len(images)} images.")

    # 2. Generate random pairs
    print(f"Generating {config.NUM_COMPARISON_PAIRS} pairs...")
    pairs = []
    
    for _ in tqdm(range(config.NUM_COMPARISON_PAIRS), desc="Creating Pairs"):
        img_a, img_b = random.sample(images, 2)
        pairs.append((img_a, img_b, letter_to_process))

    # 3. Write to CSV
    output_filename = f"pairs_{letter_to_process}.csv"
    try:
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_a', 'image_b', 'character'])
            writer.writerows(pairs)
        print(f"\nSuccessfully created '{output_filename}' with {len(pairs)} pairs.")
    except Exception as e:
        print(f"\nError writing CSV file: {e}")

if __name__ == "__main__":
    main()