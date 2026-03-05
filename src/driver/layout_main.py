import argparse
from pathlib import Path
import logging
from utils import iterate_image_paths, check_for_output

def cli():

    log = logging.getLogger("layout_detection_surya")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="Layout Detection with surya")

    parser.add_argument("--images_path",
                        type=Path,
                        required=True,
                        default=Path("dataset/bank_statments/raw/images"),
                        help="Path to directory consisting images")

    parser.add_argument("--model_for_detection",
                        type=str,
                        default="surya",
                        help="Model used for table detection")
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for inference")
    
    parser.add_argument("--layout_dir",
                        type=Path,
                        required=True,
                        help="Save layout detections in json format")
    
    parser.add_argument("--save_visualizations",
                        type=str,
                        help="Path to save visualizations of the model")
    
    args = parser.parse_args()

    # Collect the image paths from the images directory
    image_files = list(iterate_image_paths(args.images_path))
    # Resume layout detection from images that are not yet processed
    images_to_process = [p for p in image_files if not check_for_output(args.layout_dir,p)] 

    #Log the number of images that need to be processed
    log.info(f"The number of images yet to be processed are: {len(images_to_process)} out of the total number of images {len(image_files)}")



if __name__ == "__main__":
    cli()