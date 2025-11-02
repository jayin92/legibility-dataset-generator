import cv2
import numpy as np
import skimage.transform as skimg_tr
import random
import config

# --- 1. LOCAL STRUCTURAL DEFORMATION ---
def slice_and_stretch(img, stretch_factor=2.0, axis='y'):
    """
    Slices an image in three parts and stretches the middle part.
    axis='y' stretches vertically (like a giraffe's neck).
    axis='x' stretches horizontally (like a pig's body).
    """
    if axis == 'y':
        rows, cols = img.shape[:2]
        top_slice_end = int(rows * 0.25)
        bottom_slice_start = int(rows * 0.75)

        top_part = img[0:top_slice_end, :]
        middle_part = img[top_slice_end:bottom_slice_start, :]
        bottom_part = img[bottom_slice_start:rows, :]

        new_middle_height = int(middle_part.shape[0] * stretch_factor)
        if new_middle_height <= 0 or middle_part.shape[0] <= 0 or middle_part.shape[1] <= 0:
            return img

        stretched_middle = cv2.resize(middle_part,
                                    (cols, new_middle_height),
                                    interpolation=cv2.INTER_LINEAR)
        return np.vstack((top_part, stretched_middle, bottom_part))
    
    else: # axis == 'x'
        rows, cols = img.shape[:2]
        left_slice_end = int(cols * 0.25)
        right_slice_start = int(cols * 0.75)

        left_part = img[:, 0:left_slice_end]
        middle_part = img[:, left_slice_end:right_slice_start]
        right_part = img[:, right_slice_start:cols]

        new_middle_width = int(middle_part.shape[1] * stretch_factor)
        if new_middle_width <= 0 or middle_part.shape[0] <= 0 or middle_part.shape[1] <= 0:
            return img

        stretched_middle = cv2.resize(middle_part,
                                    (new_middle_width, rows),
                                    interpolation=cv2.INTER_LINEAR)
        return np.hstack((left_part, stretched_middle, right_part))

# --- 2. GLOBAL GEOMETRIC DEFORMATIONS (AFFINE/PROJECTIVE) ---
def affine_transform(img):
    """Applies a randomized affine transformation (scale, shear)."""
    rows, cols = img.shape[:2]
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
    
    dst_points = src_points.copy()
    dst_points[0,0] += np.random.uniform(-cols*0.1, cols*0.1)
    dst_points[0,1] += np.random.uniform(-rows*0.1, rows*0.1)
    dst_points[1,0] += np.random.uniform(-cols*0.2, cols*0.2)
    dst_points[1,1] += np.random.uniform(-rows*0.1, rows*0.1)
    dst_points[2,0] += np.random.uniform(-cols*0.1, cols*0.1)
    dst_points[2,1] += np.random.uniform(-rows*0.2, rows*0.2)
    
    M = cv2.getAffineTransform(src_points, dst_points)
    return cv2.warpAffine(img, M, (cols, rows), borderValue=config.BACKGROUND_COLOR)

def projective_transform(img):
    """Applies a randomized projective transformation (perspective skew)."""
    rows, cols = img.shape[:2]
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    
    dst_points = src_points.copy()
    for i in range(4):
        dst_points[i, 0] += np.random.uniform(-cols*0.2, cols*0.2)
        dst_points[i, 1] += np.random.uniform(-rows*0.2, rows*0.2)
        
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, M, (cols, rows), borderValue=config.BACKGROUND_COLOR)

# --- 3. NON-LINEAR DEFORMATION (TPS) ---
def tps_transform(img):
    """Applies a Thin-Plate Spline (TPS) warp for non-linear bending."""
    rows, cols = img.shape[:2]
    
    src_points = np.array([
        [0, 0], [cols/2, 0], [cols-1, 0],
        [0, rows/2], [cols/2, rows/2], [cols-1, rows/2],
        [0, rows-1], [cols/2, rows-1], [cols-1, rows-1]
    ])
    
    dst_points = src_points.copy()
    for i in range(1, 8): # Don't move corners
        dst_points[i, 0] += np.random.uniform(-cols*0.2, cols*0.2)
        dst_points[i, 1] += np.random.uniform(-rows*0.2, rows*0.2)
        
    tps = skimg_tr.ThinPlateSplineTransform()
    tps.estimate(src_points, dst_points)
    
    # skimage warp returns float[0,1]
    warped_img_float = skimg_tr.warp(img, tps, output_shape=(rows, cols),
                                     cval=1.0 if config.BACKGROUND_COLOR == (255,255,255) else 0.0)
    
    # Convert back to uint8
    if config.IMAGE_MODE == 'RGB':
        warped_img_uint8 = (warped_img_float * 255).astype(np.uint8)
        # skimage outputs RGB, but we work in BGR, so convert back
        return cv2.cvtColor(warped_img_uint8, cv2.COLOR_RGB2BGR)
    else:
        return (warped_img_float * 255).astype(np.uint8)


# --- 4. MASTER DEFORMATION PIPELINE ---
def apply_random_deformation(img):
    """
    Applies a random chain of deformations to an input letter image.
    Includes padding to prevent clipping during transforms.
    """
    rows, cols = img.shape[:2]
    pad = config.PRE_DEFORM_PADDING
    
    # 1. Add padding
    bg_val = [c for c in config.BACKGROUND_COLOR] # (R,G,B) -> [R,G,B]
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=bg_val)
    
    # 2. Apply random chain of transforms
    
    # 2a. Local structural stretch (sometimes)
    if random.random() < 0.3: # 30% chance
        stretch_dir = random.choice(['x', 'y'])
        stretch_val = random.uniform(1.5, 3.0)
        img_padded = slice_and_stretch(img_padded, stretch_val, stretch_dir)
    
    # 2b. Non-linear bend (most of the time)
    if random.random() < 0.7: # 70% chance
        img_padded = tps_transform(img_padded)
        
    # 2c. Final global warp (either affine or projective)
    if random.random() < 0.5:
        img_padded = affine_transform(img_padded)
    else:
        img_padded = projective_transform(img_padded)
        
    # 3. Crop back to original size
    # Find bounding box of non-white pixels to re-center
    if config.IMAGE_MODE == 'RGB':
        gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_padded
        
    # Invert (find black text)
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1]
    
    x, y, w, h = cv2.boundingRect(thresh)
    
    # If no text found, return a blank image
    if w == 0 or h == 0:
        return np.full((rows, cols, 3 if config.IMAGE_MODE == 'RGB' else 1), 
                       255, dtype=np.uint8)
    
    cropped = img_padded[y:y+h, x:x+w]
    
    # 4. Resize back to the original 512x512, preserving aspect ratio
    
    # Create a 512x512 canvas
    final_canvas = np.full((rows, cols, 3 if config.IMAGE_MODE == 'RGB' else 1), 
                           255, dtype=np.uint8)
    if config.IMAGE_MODE == 'RGB':
        final_canvas[:] = config.BACKGROUND_COLOR
    
    # Scale the cropped image to fit
    max_dim = max(w, h)
    scale = config.IMAGE_SIZE[0] / max_dim
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w <= 0 or new_h <= 0:
        return final_canvas # Return blank canvas
        
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Paste it in the center
    paste_x = (config.IMAGE_SIZE[0] - new_w) // 2
    paste_y = (config.IMAGE_SIZE[1] - new_h) // 2
    
    final_canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
    
    return final_canvas

