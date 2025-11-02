# GEMINI.md

## Project Overview

This project, `legibility-dataset-generator`, is a Python-based tool for creating large-scale datasets of deformed characters. The primary purpose is to generate training data for Vision Language Models (VLMs) to learn about character legibility.

The generation pipeline consists of the following steps:
1.  **Character Rendering:** Characters are rendered using a variety of fonts from the `assets/fonts` directory.
2.  **Deformation:** A series of random deformations are applied to the rendered character images to create variations in legibility.
3.  **Image Saving:** The final deformed images are saved to the `generated_dataset` directory, organized into subdirectories for each character.
4.  **Pair Generation:** A CSV file (`pairs.csv`) is created containing pairs of image filenames, which can be used for comparison-based training tasks.

The core logic is orchestrated through a pipeline that processes jobs in parallel for efficiency. Configuration for the dataset generation process is centralized in `config.py`.

## Key Technologies

*   **Language:** Python
*   **Dependency Management:** uv
*   **Core Libraries:**
    *   `opencv-contrib-python`: For image processing and deformations.
    *   `Pillow`: For image creation and text rendering.
    *   `numpy`: For numerical operations on image data.
    *   `scikit-image`: For various image processing tasks.
    *   `tqdm`: For displaying progress bars.

## Building and Running

### 1. Create a Virtual Environment and Install Dependencies

This project uses `uv` for dependency management. First, create a virtual environment:

```bash
uv venv
```

Then, activate the environment and install the dependencies from `pyproject.toml`:

```bash
source .venv/bin/activate
uv pip install -e .
```

### 2. Run the Dataset Generation

To generate the dataset, execute the `generate_dataset.py` script. This script orchestrates the rendering, deformation, and saving of character images using multiprocessing.

```bash
python generate_dataset.py
```

### 3. Generate Comparison Pairs

To generate the comparison pairs CSV file, run the `generate_pairs.py` script after the dataset has been generated.

```bash
python generate_pairs.py
```

## Development Conventions

*   **Configuration:** All major settings for dataset generation (e.g., characters, image dimensions, number of images) are managed in the `config.py` file.
*   **Modularity:** The codebase is organized into distinct modules, each with a clear responsibility:
    *   `font_loader.py`: Handles loading fonts from the assets directory.
    *   `renderer.py`: Responsible for rendering initial character images.
    *   `deformations.py`: Contains functions to apply various deformations to images.
    *   `pipeline.py`: Defines the main processing job for a single image.
*   **Parallelism:** The processing pipeline is designed to be run in parallel using multiprocessing to speed up dataset generation.
