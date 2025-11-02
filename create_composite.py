
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import config

def create_side_by_side_image_legacy(image_a_path, image_b_path, output_path=None):
    """The original basic function to create a side-by-side image."""
    try:
        img_a = Image.open(image_a_path)
        img_b = Image.open(image_b_path)
    except FileNotFoundError:
        print(f"Error: One or both image files not found: {image_a_path}, {image_b_path}")
        return None
    except Exception as e:
        print(f"Error opening images: {e}")
        return None

    if img_a.mode != 'RGB':
        img_a = img_a.convert('RGB')
    if img_b.mode != 'RGB':
        img_b = img_b.convert('RGB')

    width, height = img_a.size
    composite_width = width * 2
    header_height = 50
    composite_height = height + header_height
    
    composite_img = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
    
    composite_img.paste(img_a, (0, header_height))
    composite_img.paste(img_b, (width, header_height))
    
    draw = ImageDraw.Draw(composite_img)
    try:
        font_path = os.path.join("assets", "fonts", "Roboto-Regular.ttf")
        font = ImageFont.truetype(font_path, 48)
    except IOError:
        print(f"Warning: Font not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    text_a_bbox = draw.textbbox((0,0), "Image A", font=font)
    text_a_width = text_a_bbox[2] - text_a_bbox[0]
    text_a_height = text_a_bbox[3] - text_a_bbox[1]

    text_b_bbox = draw.textbbox((0,0), "Image B", font=font)
    text_b_width = text_b_bbox[2] - text_b_bbox[0]

    draw.text((width / 2 - text_a_width / 2, (header_height - text_a_height) / 2), "Image A", fill="black", font=font)
    draw.text((width + width / 2 - text_b_width / 2, (header_height - text_a_height) / 2), "Image B", fill="black", font=font)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite_img.save(output_path)
        print(f"Composite image saved to {output_path}")
        
    return composite_img

def create_side_by_side_image(image_a_path, image_b_path, output_path=None, legacy=False):
    """
    Creates a side-by-side image with labels, with a flag to use the legacy version.
    Saves the composite image to output_path if provided.
    """
    if legacy:
        return create_side_by_side_image_legacy(image_a_path, image_b_path, output_path)

    try:
        img_a = Image.open(image_a_path)
        img_b = Image.open(image_b_path)
    except FileNotFoundError:
        print(f"Error: One or both image files not found: {image_a_path}, {image_b_path}")
        return None
    except Exception as e:
        print(f"Error opening images: {e}")
        return None

    if img_a.mode != 'RGB':
        img_a = img_a.convert('RGB')
    if img_b.mode != 'RGB':
        img_b = img_b.convert('RGB')

    base_width, base_height = img_a.size
    
    padding = config.COMPOSITE_PADDING
    border = config.COMPOSITE_BORDER_WIDTH
    divider = config.COMPOSITE_DIVIDER_WIDTH
    header_h = config.COMPOSITE_HEADER_HEIGHT
    
    content_h = base_height + (2 * padding)
    content_w = base_width + (2 * padding)
    
    composite_w = (2 * content_w) + divider
    composite_h = content_h + header_h

    composite_img = Image.new('RGB', (composite_w, composite_h), config.COMPOSITE_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(composite_img)

    draw.rectangle([0, 0, composite_w, header_h], fill=config.COMPOSITE_HEADER_COLOR)

    border_x0_a = padding - border
    border_y0_a = header_h + padding - border
    border_x1_a = padding + base_width + border
    border_y1_a = header_h + padding + base_height + border
    draw.rectangle([border_x0_a, border_y0_a, border_x1_a, border_y1_a], fill=config.COMPOSITE_BORDER_COLOR)
    composite_img.paste(img_a, (padding, header_h + padding))

    border_x0_b = content_w + divider + padding - border
    border_y0_b = header_h + padding - border
    border_x1_b = content_w + divider + padding + base_width + border
    border_y1_b = header_h + padding + base_height + border
    draw.rectangle([border_x0_b, border_y0_b, border_x1_b, border_y1_b], fill=config.COMPOSITE_BORDER_COLOR)
    composite_img.paste(img_b, (content_w + divider + padding, header_h + padding))

    divider_x = content_w + (divider // 2)
    draw.line([divider_x, header_h, divider_x, composite_h], fill=config.COMPOSITE_DIVIDER_COLOR, width=divider)

    try:
        font_path = os.path.join("assets", "fonts", "Roboto-Regular.ttf")
        font_size = int(header_h * 0.6)
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    text_a = "Image A"
    text_a_bbox = draw.textbbox((0, 0), text_a, font=font)
    text_a_width = text_a_bbox[2] - text_a_bbox[0]
    text_a_height = text_a_bbox[3] - text_a_bbox[1]
    text_a_x = (content_w / 2) - (text_a_width / 2)
    text_a_y = (header_h / 2) - (text_a_height / 2)
    draw.text((text_a_x, text_a_y), text_a, fill=config.COMPOSITE_TEXT_COLOR, font=font)

    text_b = "Image B"
    text_b_bbox = draw.textbbox((0, 0), text_b, font=font)
    text_b_width = text_b_bbox[2] - text_b_bbox[0]
    text_b_height = text_b_bbox[3] - text_b_bbox[1]
    text_b_x = content_w + divider + (content_w / 2) - (text_b_width / 2)
    text_b_y = (header_h / 2) - (text_b_height / 2)
    draw.text((text_b_x, text_b_y), text_b, fill=config.COMPOSITE_TEXT_COLOR, font=font)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite_img.save(output_path)
        print(f"Composite image saved to {output_path}")
    
    return composite_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a side-by-side composite image.")
    parser.add_argument("image_a", help="Path to the first image.")
    parser.add_argument("image_b", help="Path to the second image.")
    parser.add_argument("output", help="Path to save the composite image.")
    parser.add_argument("--legacy-composite", action="store_true", help="Use the legacy composite image creation logic.")
    args = parser.parse_args()

    if not os.path.exists(args.image_a) or not os.path.exists(args.image_b):
        print("Error: One or both input images not found.")
    else:
        create_side_by_side_image(args.image_a, args.image_b, args.output, legacy=args.legacy_composite)
