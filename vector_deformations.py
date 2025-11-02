import random
import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from skimage.morphology import skeletonize
import config
import renderer

def stroke_thickness_transform(points):
    """
    Applies the stroke thickness transformation to a list of skeleton points.
    Returns a list of points and their corresponding radiuses.
    """
    n = len(points)
    if n < 2:
        return points, [1] * n

    # As per paper
    ns = random.choice([4, 6, 8, 10])
    # 10 is an assumed default stroke thickness, can be made configurable
    base_thickness = random.uniform(0.8, 1.5) * 10 
    
    # Define spline control points for the radiuses
    x_control = np.linspace(0, n - 1, ns)
    y_control = []
    for _ in range(ns):
        variation = random.uniform(-0.5, 0.5) * base_thickness
        y_control.append(base_thickness + variation)

    # Create cubic spline to smoothly interpolate radiuses
    cs = CubicSpline(x_control, y_control)

    # Generate radiuses for all points along the skeleton
    x_all = np.arange(n)
    radiuses = cs(x_all)
    
    # Clip radiuses to be non-negative
    radiuses[radiuses < 1] = 1
    
    return points, radiuses

def render_disks(points, radiuses):
    """
    Renders a character from its skeleton points and stroke radiuses.
    The canvas is the final configured image size.
    """
    canvas = np.full((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3 if config.IMAGE_MODE == 'RGB' else 1), 
                     255, dtype=np.uint8)
    
    if config.IMAGE_MODE == 'RGB':
        bg_color_bgr = (config.BACKGROUND_COLOR[2], config.BACKGROUND_COLOR[1], config.BACKGROUND_COLOR[0])
        canvas[:] = bg_color_bgr
        color = (config.TEXT_COLOR[2], config.TEXT_COLOR[1], config.TEXT_COLOR[0])
    else:
        canvas[:] = 255
        color = 0

    # The points are already scaled to the final canvas size
    for i in range(len(points)):
        center = (int(points[i][0]), int(points[i][1]))
        radius = int(radiuses[i])
        if radius > 0:
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
            
    return canvas

def generate_vector_deformed_image(char, font_path, font_index):
    """
    The main function to generate an image using the SKELETON-based vector deformation.
    This image is NOT padded and is ready for the raster deformation pipeline.
    """
    # 1. Render the character to a standard raster image first.
    base_img = renderer.render_character(char, font_path, font_index)
    if base_img is None:
        return None

    # 2. Prepare for skeletonization (convert to binary).
    if len(base_img.shape) == 3:
        gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = base_img
    
    # Skeletonization in scikit-image expects the shape to be `True` or `1`.
    # Our text is black (0) on a white (255) background.
    binary_img = gray < 128

    # 3. Perform skeletonization.
    skeleton = skeletonize(binary_img)

    # 4. Extract points from the skeleton.
    # np.argwhere returns (row, col), which corresponds to (y, x).
    points_yx = np.argwhere(skeleton)
    if points_yx.shape[0] == 0:
        return None # No skeleton found
    
    # Sort points to create a somewhat ordered path for the spline.
    # This is an approximation that traces from top-to-bottom, left-to-right.
    # It's necessary because the spline requires ordered input.
    sorted_points_yx = sorted(points_yx, key=lambda p: (p[0], p[1]))
    
    # Convert (y, x) tuples to (x, y) for drawing
    final_points_xy = [(p[1], p[0]) for p in sorted_points_yx]

    # 5. Apply stroke thickness transformation to the skeleton points.
    points_with_thickness, radiuses = stroke_thickness_transform(final_points_xy)
    
    # 6. Render the final image by drawing disks.
    image = render_disks(points_with_thickness, radiuses)
    
    return image