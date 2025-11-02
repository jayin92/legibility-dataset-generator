import os
import cv2
import random
import config
import renderer
import deformations
import vector_deformations

def process_job(job_args):
    """
    A single worker function for the multiprocessing pool.
    Takes one job, processes it, and saves the file.
    """
    # Unpack arguments
    char, font_path, font_index, font_name, job_id, total_jobs = job_args
    
    try:
        # 1. Generate the base character image
        base_img = None
        # Randomly choose to use vector or raster deformation for base image
        if random.random() < config.USE_VECTOR_DEFORMATION_PROB:
            base_img = vector_deformations.generate_vector_deformed_image(char, font_path, font_index)
            # Save for debugging
            # cv2.imwrite(f"debug/debug_vector_{char}_{font_name}_{job_id:05d}.png", base_img)
            

        # If vector deformation failed or was not chosen, use the original raster renderer
        if base_img is None:
            base_img = renderer.render_character(char, font_path, font_index)

        if base_img is None:
            return f"Failed to render {char} from {font_name} using any method"

        # 2. Apply random raster deformations (e.g., warp, perspective)
        deformed_img = deformations.apply_random_deformation(base_img)

        # 3. Create output path and save
        # e.g., generated_dataset/a_upper/a_Arial_00001.png
        char_for_path = char
        if 'a' <= char <= 'z':
            dir_name = f"{char_for_path}_lower"
        elif 'A' <= char <= 'Z':
            char_for_path = char.lower()
            dir_name = f"{char_for_path}_upper"
        else:
            dir_name = char_for_path
            
        char_dir = os.path.join(config.OUTPUT_DIR, dir_name)
        if not os.path.exists(char_dir):
            try:
                os.makedirs(char_dir, exist_ok=True)
            except FileExistsError:
                pass # Race condition in multiprocessing

        filename = f"{char_for_path}_{font_name}_{job_id:05d}.png"
        output_path = os.path.join(char_dir, filename)
        
        cv2.imwrite(output_path, deformed_img)

        # Log progress for one of the processes
        if job_id % (total_jobs // 100) == 0:
             print(f"Progress: Processed {job_id}/{total_jobs}...")

        return f"OK: {output_path}"
        
    except Exception as e:
        return f"ERROR processing {char} from {font_name}: {e}"

