import os
import cv2
import config
import renderer
import deformations

def process_job(job_args):
    """
    A single worker function for the multiprocessing pool.
    Takes one job, processes it, and saves the file.
    """
    # Unpack arguments
    char, font_path, font_index, font_name, job_id, total_jobs = job_args
    
    try:
        # 1. Render the base character
        base_img = renderer.render_character(char, font_path, font_index)
        if base_img is None:
            return f"Failed to render {char} from {font_name}"

        # 2. Apply random deformations
        deformed_img = deformations.apply_random_deformation(base_img)

        # 3. Create output path and save
        # e.g., generated_dataset/A/A_Arial_00001.png
        char_dir = os.path.join(config.OUTPUT_DIR, char)
        if not os.path.exists(char_dir):
            try:
                os.makedirs(char_dir, exist_ok=True)
            except FileExistsError:
                pass # Race condition in multiprocessing

        filename = f"{char}_{font_name}_{job_id:05d}.png"
        output_path = os.path.join(char_dir, filename)
        
        cv2.imwrite(output_path, deformed_img)

        # Log progress for one of the processes
        if job_id % (total_jobs // 100) == 0:
             print(f"Progress: Processed {job_id}/{total_jobs}...")

        return f"OK: {output_path}"
        
    except Exception as e:
        return f"ERROR processing {char} from {font_name}: {e}"

