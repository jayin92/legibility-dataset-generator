import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import config

def render_character(char, font_path, font_index=0):
    """
    Renders a single character onto a 512x512 canvas, ensuring it fills the image
    as much as possible while maintaining aspect ratio and being centered.
    """
    target_width, target_height = config.IMAGE_SIZE

    # Step 1: Find an appropriate font size to make the character fill the canvas.
    # We'll use a temporary, very large canvas and a placeholder font size to measure.
    temp_font_size = 1000 # Large size to ensure accurate measurement
    
    try:
        # Load the font for measurement
        font_measure = ImageFont.truetype(font_path, size=temp_font_size, index=font_index)
    except (IOError, OSError) as e:
        print(f"Error loading font {font_path} at index {font_index}: {e}")
        return None

    try:
        # Get bounding box [left, top, right, bottom]
        bbox = font_measure.getbbox(char)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
    except (TypeError, ValueError):
        # Some fonts may fail on some chars
        return None

    if char_width == 0 or char_height == 0:
        # Cannot render empty or invisible characters
        return None

    # Calculate the scale factor needed to fit the character's bbox into the target size
    width_scale = target_width / char_width
    height_scale = target_height / char_height
    
    # Use the smaller scale to fit entirely within the image
    scale = min(width_scale, height_scale)
    
    # Adjust font size based on this scale
    final_font_size = int(temp_font_size * scale)
    if final_font_size <= 0: # Ensure font size is at least 1
        final_font_size = 1
        
    try:
        # Load the font with the calculated final size
        font_final = ImageFont.truetype(font_path, size=final_font_size, index=font_index)
    except (IOError, OSError) as e:
        print(f"Error loading font {font_path} at index {font_index}: {e}")
        return None

    # Step 2: Render onto the actual target canvas using the final font size.
    img = Image.new(config.IMAGE_MODE, config.IMAGE_SIZE, config.BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Get the bbox for the *finally sized* character
    try:
        final_bbox = font_final.getbbox(char)
        final_char_width = final_bbox[2] - final_bbox[0]
        final_char_height = final_bbox[3] - final_bbox[1]
    except (TypeError, ValueError):
        return None # Should not happen if previous bbox was fine

    # Calculate precise position to center the character's bounding box
    # We want final_char_width to be centered in target_width
    # And final_char_height to be centered in target_height
    center_x = (target_width - final_char_width) / 2
    center_y = (target_height - final_char_height) / 2
    
    # The draw.text origin is relative to the font's internal baseline,
    # so we need to offset by the bbox's top-left corner.
    draw_x = center_x - final_bbox[0]
    draw_y = center_y - final_bbox[1]
    
    draw.text((draw_x, draw_y), char, font=font_final, fill=config.TEXT_COLOR)

    # Step 3: Convert PIL Image to NumPy array for OpenCV
    img_np = np.array(img)
    
    if config.IMAGE_MODE == 'RGB':
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr
    else: # Grayscale
        return img_np

