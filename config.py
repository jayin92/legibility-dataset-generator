# --- Dataset Settings ---
# Characters to generate.
# CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Total number of images to generate PER CHARACTER.
# e.g., 1000 * len(CHARACTERS) = 62,000 total images.
IMAGES_PER_CHARACTER = 500

# --- Directory Settings ---
FONT_DIR = "assets/fonts"
OUTPUT_DIR = "data/distort_letter"
USE_CPU_RATIO = 0.9 # Use 90% of available CPU cores

# --- Image Settings ---
IMAGE_SIZE = (512, 512) # (width, height)   
BACKGROUND_COLOR = (255, 255, 255) # White
TEXT_COLOR = (0, 0, 0) # Black
IMAGE_MODE = "RGB" # "RGB" or "L" (grayscale)

# --- Deformation Settings ---
# Probability of using the new vector-based stroke thickness deformation
USE_VECTOR_DEFORMATION_PROB = 0.25 # 25% chance

# Padding added before deformation to prevent clipping.
# This should be a large value.
PRE_DEFORM_PADDING = 256 # pixels

# --- Pair Generation Settings ---
# The number of comparison pairs to generate for the VLM.
# e.g., 50,000 pairs like ('a_001.png', 'a_523.png')
NUM_COMPARISON_PAIRS = 10000
PAIR_CSV_FILE = "pairs.csv"

# --- Composite Image Settings ---
COMPOSITE_HEADER_HEIGHT = 60
COMPOSITE_PADDING = 20
COMPOSITE_BORDER_WIDTH = 2
COMPOSITE_DIVIDER_WIDTH = 2
COMPOSITE_BACKGROUND_COLOR = (255, 255, 255) # White
COMPOSITE_HEADER_COLOR = (240, 240, 240) # Light Grey
COMPOSITE_BORDER_COLOR = (192, 192, 192) # Grey
COMPOSITE_DIVIDER_COLOR = (192, 192, 192) # Grey
COMPOSITE_TEXT_COLOR = (0, 0, 0) # Black
