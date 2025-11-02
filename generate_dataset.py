import os
import random
import multiprocessing
from tqdm import tqdm
import config
import font_loader
import pipeline

def main():
    print("--- Legibility Dataset Generator ---")
    
    # 1. Create output directory
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Created output directory: {config.OUTPUT_DIR}")

    # 2. Load all available fonts
    fonts = font_loader.load_fonts(config.FONT_DIR)
    if not fonts:
        return

    # 3. Build the job list
    print("Building job list...")
    job_list = []
    total_images = 0
    
    for char in config.CHARACTERS:
        for i in range(config.IMAGES_PER_CHARACTER):
            # Pick a random font for this job
            font_path, font_index, font_name = random.choice(fonts)
            
            # (char, font_path, font_index, font_name, job_id, total_jobs)
            job_args = (
                char, 
                font_path, 
                font_index, 
                font_name,
                i + 1, # job_id
                config.IMAGES_PER_CHARACTER * len(config.CHARACTERS)
            )
            job_list.append(job_args)
            total_images += 1
            
    print(f"Total jobs created: {len(job_list)}")
    random.shuffle(job_list) # Shuffle to distribute font types

    # 4. Execute jobs in parallel
    num_cpus = multiprocessing.cpu_count()
    print(f"Starting dataset generation on {num_cpus} CPU cores...")
    
    # Use pool.imap_unordered for efficiency and tqdm for progress
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = list(tqdm(pool.imap_unordered(pipeline.process_job, job_list), 
                            total=len(job_list), 
                            desc="Generating Images"))

    print("\n--- Generation Complete ---")
    ok_count = sum(1 for r in results if r.startswith("OK"))
    err_count = len(results) - ok_count
    print(f"Successfully generated: {ok_count} images")
    print(f"Failed jobs: {err_count}")
    if err_count > 0:
        print("First 10 errors:")
        for r in results:
            if r.startswith("ERROR"):
                print(r)
                err_count -= 1
                if err_count <= 0:
                    break

if __name__ == "__main__":
    main()

