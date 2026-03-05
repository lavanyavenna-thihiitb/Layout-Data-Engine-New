from pathlib import Path
from typing import Iterator

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