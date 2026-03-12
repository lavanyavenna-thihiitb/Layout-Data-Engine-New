import argparse
from pathlib import Path
import logging
from layout_detection_new.utils import *
from layout_detection_new.detectors import SuryaModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("layout_detection_surya")

def cli():

    parser = argparse.ArgumentParser(description="Layout Detection with surya")

    parser.add_argument("--root",
                        type=Path,
                        required=True,
                        help="Path to directory consisting images like - dataset/bank_statments/raw/images")

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
                        default=Path("json_outputs_ids/"),
                        help="Save layout detections in json format.")
    
    parser.add_argument("--save_visualizations",
                        type=Path,
                        default=Path("visual_outputs_ids/"),
                        help="Path to save visualizations of the model")
    
    args = parser.parse_args()

    # Collect the image paths from the images directory
    image_files = list(iterate_image_paths(args.root))
    # Resume layout detection from images that are not yet processed
    images_to_process = [p for p in image_files if not check_for_output(args.layout_dir,p)] 

    #Log the number of images that need to be processed
    log.info(f"The number of images yet to be processed are: {len(images_to_process)} out of the total number of images {len(image_files)}")

    if not images_to_process:
        log.info(f"No new images to process!")
        return
    
    batches = create_batches(images_to_process, args.batch_size)
    log.info("Batches created, starting layout detection......")

    if args.model_for_detection == "surya":
        model = SuryaModel()
        model.load_model()
        model.layout_detection(batches, args.layout_dir, args.save_visualizations)
    
    log.info(f"Successfully finished processing images in {args.root}")


if __name__ == "__main__":
    print(f"Initiating Layout Detection ......")
    log.info("Initiating Layout detection........")
    cli()