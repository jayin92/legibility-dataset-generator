import os
import config
from PIL import ImageFont

def load_fonts(font_dir):
    """
    Scans the font directory and loads all .ttf and .ttc files.
    
    A .ttc file is a "TrueType Collection" and can contain
    multiple fonts. We try to load the first few indices from them.
    
    Returns:
        list: A list of (font_path, font_index, font_name) tuples.
    """
    font_files = []
    print(f"Scanning for fonts in '{font_dir}'...")
    
    if not os.path.exists(font_dir):
        print(f"Error: Font directory not found: {font_dir}")
        print("Please create it and add .ttf or .ttc files.")
        return []

    for filename in os.listdir(font_dir):
        file_path = os.path.join(font_dir, filename)
        file_lower = filename.lower()
        base_name = os.path.splitext(filename)[0]

        if file_lower.endswith('.ttf'):
            # Standard TrueType Font
            font_files.append((file_path, 0, base_name))
            
        elif file_lower.endswith('.ttc'):
            # TrueType Collection - can have multiple fonts
            # We'll try loading the first 10 indices
            for i in range(10):
                try:
                    # Test load to see if index is valid
                    ImageFont.truetype(file_path, size=10, index=i)
                    font_files.append((file_path, i, f"{base_name}_index{i}"))
                except (IOError, OSError):
                    # Stop if we hit an invalid index
                    break
                    
    print(f"Found {len(font_files)} loadable fonts.")
    return font_files

