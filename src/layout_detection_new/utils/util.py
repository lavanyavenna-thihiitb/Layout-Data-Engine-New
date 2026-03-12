from pathlib import Path
from typing import Iterator

def create_batches(images: list[Path], batch_size: int) -> Iterator[list[Path]]:
    """
    Splits a list of image paths into batches.

    Args:
        images: List of image paths to process
        batch_size: Number of images per batch

    Yields:
        List[Path]: Batch of image paths
    """

    for i in range(0, len(images), batch_size):
        yield images[i:i+batch_size]

def iterate_image_paths(images_dir:Path) -> Iterator[Path]:
    """
    Iterate over all image files in a directory.

    Args:
        images_dir (Path): Path to directory containing images

    Yields:
        Path: Path to each image file
    """
    image_extensions = {".jpg", ".jpeg", ".png"}

    for file_path in images_dir.iterdir():
        if file_path.suffix.lower() in image_extensions:
            yield file_path

def check_for_output(output_dir:Path,image_path:Path) -> bool:
    """
    The output directory is going to be in the structure:
    
    json_outputs
    ├── bank_statments
    │   └── outputs
    │       ├── json_of_images --> image_base_name.json 
    │       └── json_of_pdf
    ├── itr_forms
    │   └── outputs
    │       ├── json_of_images
    │       └── json_of_pdf

    Args: 
        output_dir: The output directory that contains .json file
        image_path: base name of the image path

    Returns:
        true: if output for that image exists
        False: if ouput for that image doesn't exists

    """

    #Extract image base name
    image_base_name = image_path.stem

    #Expected json filename
    json_file = output_dir / f"{image_base_name}.json"

    #Check if json exists
    return json_file.exists()
    

def get_visual_directory_for_json(input_dir:list, layout_dir:Path) -> list[Path]:
    
    """
    Generates JSON ouput paths for a batch of input images based on dataset type

    Args:
        input_batch: List of image paths from the path - dataset/bank_statmeents/raw/images/.jpg
        layout_dir: Base path of where to store them

    Returns:
    list[Path]: JSON files for the corresponding images

    """

    json_paths = []

    for image_paths in input_dir:

        #image path is dataset/bank_statmeents/raw/images/.jpg
        dataset_name = image_paths.parts[-4]

        json_dir = layout_dir / dataset_name / "outputs" / "json_of_images"

        json_dir.mkdir(parents=True, exist_ok=True)

        json_file = json_dir / f"{image_paths.stem}.json"

        json_paths.append(json_file)

    return json_paths
