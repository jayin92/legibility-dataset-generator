# --- Dataset Settings ---
# Characters to generate.
# CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Total number of images to generate PER CHARACTER.
# e.g., 1000 * len(CHARACTERS) = 62,000 total images.
IMAGES_PER_CHARACTER = 500

# --- Directory Settings ---
FONT_DIR = "assets/fonts"
OUTPUT_DIR = "outputs/generated_dataset"

# --- Image Settings ---
IMAGE_SIZE = (512, 512) # (width, height)
BACKGROUND_COLOR = (255, 255, 255) # White
TEXT_COLOR = (0, 0, 0) # Black
IMAGE_MODE = "RGB" # "RGB" or "L" (grayscale)

# --- Deformation Settings ---
# Padding added before deformation to prevent clipping.
# This should be a large value.
PRE_DEFORM_PADDING = 256 # pixels

# --- Pair Generation Settings ---
# The number of comparison pairs to generate for the VLM.
# e.g., 50,000 pairs like ('a_001.png', 'a_523.png')
NUM_COMPARISON_PAIRS = 50000
PAIR_CSV_FILE = "pairs.csv"

