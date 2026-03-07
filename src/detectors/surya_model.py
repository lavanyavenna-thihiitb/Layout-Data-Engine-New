import os
import json
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings
import logging
from utils import get_visual_directory_for_json

class SuryaModel:

    def __init__(self) -> None:
        self.log = logging.getLogger("layout_detection_surya")
        self.log.info("Initializing Surya Model!")
        self.foundation_predictor = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        self.layout_predictor = LayoutPredictor(self.foundation_predictor)

    def layout_detection(self, input_batch_images, batch_size, layout_directory) -> None:

        try:
            for batch_id, batch in enumerate(input_batch_images, start=1):
                self.log.info(f"Processing batch {batch_id} of size {len(batch)}")
                
                visual_output_dir = get_visual_directory_for_json(batch, layout_directory)

                if not visual_output_dir:
                    self.log.error(f"No output directory for visualization exists")
                    return
                
                batch_images = [Image.open(each_image_path) for each_image_path in batch]
                batch_results = self.layout_predictor(batch_images)
                
                for result in batch_results:
                    
                    for bbox in result.bboxes:
                        if bbox.label=='Table':
                            coords=bbox.bbox
                            

            

