import os
import random
import csv
import config
from tqdm import tqdm

def main():
    print("--- VLM Pairwise Comparison List Generator ---")
    
    if not os.path.exists(config.OUTPUT_DIR):
        print(f"Error: Output directory not found: {config.OUTPUT_DIR}")
        print("Please run 'generate_dataset.py' first.")
        return

    # 1. Scan output directory and group images by character
    print("Scanning generated images...")
    images_by_char = {char: [] for char in config.CHARACTERS}
    
    for char in config.CHARACTERS:
        if 'a' <= char <= 'z':
            dir_name = f"{char}_lower"
        elif 'A' <= char <= 'Z':
            char_ = char.lower()
            dir_name = f"{char_}_upper"
        else:
            dir_name = char

        char_dir = os.path.join(config.OUTPUT_DIR, dir_name)
        if not os.path.exists(char_dir):
            print(f"Warning: No images found for character '{char}'")
            continue
            
        for img_name in os.listdir(char_dir):
            if img_name.endswith('.png'):
                # Store the *relative path* for the CSV
                images_by_char[char].append(os.path.join(dir_name, img_name))
    
    # Filter out characters with less than 2 images
    valid_chars = [char for char, imgs in images_by_char.items() if len(imgs) >= 2]
    if not valid_chars:
        print("Error: Not enough images found to create pairs.")
        return
        
    print(f"Found images for {len(valid_chars)} characters.")

    # 2. Generate random pairs
    print(f"Generating {config.NUM_COMPARISON_PAIRS} pairs...")
    pairs = []
    
    for _ in tqdm(range(config.NUM_COMPARISON_PAIRS), desc="Creating Pairs"):
        # Pick a random character
        char = random.choice(valid_chars)
        
        # Sample two different images of that character
        img_a, img_b = random.sample(images_by_char[char], 2)
        
        pairs.append((img_a, img_b, char))

    # 3. Write to CSV
    try:
        with open(config.PAIR_CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_a', 'image_b', 'character'])
            writer.writerows(pairs)
        print(f"\nSuccessfully created '{config.PAIR_CSV_FILE}' with {len(pairs)} pairs.")
    except Exception as e:
        print(f"\nError writing CSV file: {e}")

if __name__ == "__main__":
    main()

