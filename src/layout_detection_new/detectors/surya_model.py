import os
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings
import logging
from pathlib import Path
from typing import Iterator
from layout_detection_new.utils import visualize, load_label_colors, get_visual_directory_for_json

#Label config path
LABEL_CONFIG_PATH = Path("src/layout_detection_new/config/label_config.yaml")

class SuryaModel:

    def __init__(self) -> None:
        self.log = logging.getLogger("layout_detection_surya")
        self.label_colors = load_label_colors(LABEL_CONFIG_PATH)

    def load_model(self) -> None:

        """
        Loads the FoundationPredictor and LayoutPredictor models.
        Call this explicityle before running layout_detection.

        Args: None
        returns: None 
        """

        self.log.info("Loading Surya Model, this may take a while...")
        self.foundation_predictor = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        self.log.info("FoundationPredictor Loaded!")
        self.layout_predictor = LayoutPredictor(self.foundation_predictor)
        self.log.info("LayoutPredictor loaded! Model is ready. ")


    def layout_detection(self, input_batch_images:Iterator[list[Path]], layout_directory:Path, visual_dir: Path | None) -> None:
        import json

        """
            Performs layout detection on batches of images and saves the results as JSON files.

            For each batch of images:
            1. Loads the images.
            2. Runs the layout detection model.
            3. Extracts detected layout elements (e.g., Text, Table, Caption, Picture, etc.).
            4. Saves the bounding box coordinates and labels to a corresponding JSON file.

            The JSON output for each image contains:
                - image_name
                - detections: list of detected layout elements with labels and bounding boxes.

            Args:
                input_batch_images: Iterator or list containing batches of image paths.
                layout_directory: Base directory where the JSON layout outputs will be saved.

            Returns:
                None
        """

        try:
            for batch_id, batch in enumerate(input_batch_images, start=1):

                self.log.info(f"Processing batch {batch_id} of size {len(batch)}")
                
                json_paths = get_visual_directory_for_json(batch, layout_directory)

                if not json_paths:
                    self.log.error(f"No output directory for visualization exists")
                    return
                
                #Load images
                batch_images = [Image.open(each_image_path).convert("RGB") for each_image_path in batch]

                #Run layout prediction
                batch_results = self.layout_predictor(batch_images)
                
                #Process predictions
                for image_path, result, json_path in zip(batch, batch_results, json_paths):

                    image_layout_data = [
                        {
                            "id": idx,
                            "label": bbox.label,
                            "bbox": bbox.bbox
                        }
                        for idx, bbox in enumerate(result.bboxes, start=1)
                    ] 

                    with open(json_path,"w") as f:
                        json.dump(
                            {
                                "image_name": image_path.name,
                                "detections": image_layout_data
                            },
                            f,
                            indent=4
                        )

                    self.log.info(f"Saved layout JSON: {json_path}")

                    if visual_dir:
                        visualize(image_path, json_path.parent, visual_dir, self.label_colors)

        except Exception as e:
            self.log.exception(f"Layout detection failed: {e}")