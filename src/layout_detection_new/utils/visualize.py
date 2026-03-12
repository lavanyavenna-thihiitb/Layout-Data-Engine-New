import json
import logging
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def get_visual_path(layout_dir:Path, visual_dir: Path) -> Path:
    """
    Mirrors the JSON output directory structure under visual_dir.

    For example:
        layout_dir : json_outputs/bank_statements/outputs/json_of_images
        visual_dir : visual_outputs/bank_statements/outputs/visual_of_images

        returns: visual_outputs/banks_statments/outputs/visual_of_images/page_01.jpg

    Args:
        layout_dir : Path where the json of images are
        visual_dir : Root visual output directory. 

    Returns:
        Full output path for the visualization file (same extension as the source image)
    """

    relative_parts = layout_dir.parts[1:]
    mirrored_dir = visual_dir / Path(*relative_parts)
    mirrored_dir.mkdir(parents=True, exist_ok=True)
    return mirrored_dir 

def load_label_colors(config_path: Path) -> dict[str, str]:
    """
    Loads label-to-color mapping from label_config.yaml.

    Args:
        config_path: Path to label_config.yaml

    Returns:
        dictionary with key as label and value as the color for the label
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("label_colors", {})
    except FileNotFoundError:
        log.warning(f"Label_config.yaml is not found at {config_path}, using empty color map")
        return {}
    except yaml.YAMLError as e:
        log.error(f"Failed to parse label_config.yaml: {e}")
        return {}

def draw_bboxes_on_image(image: Image.Image, detections: dict, label_colors: dict) -> Image.Image:
    """
    Draws colored bounding boxes and label tags onto a PIL image.

    Args:
        image:        Original PIL image.
        detections:   List of dicts with keys 'label' and 'bbox' ([x1, y1, x2, y2]).
        label_colors: Mapping of label -> hex color string.

    Returns:
        Annotated copy of the image.

    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=8)
    except IOError:
        font = ImageFont.load_default()

    default_color = label_colors.get("default", "#AAAAAA")

    for det in detections:
        try:
            id = str(det["id"])
            label = det["label"]
            x1, y1, x2, y2 = det["bbox"]
            color_hex = label_colors.get(label, default_color)
            color_rgb = ImageColor.getrgb(color_hex)

            # Semi-transparent fill
            draw.rectangle([x1, y1, x2, y2], fill=(*color_rgb,30))
            #solid outline
            draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=2)

            #Label tag just above the box - clamp to image top if needed
            tag_y = max(0, y1-8)
            text_box = draw.textbbox((x1, tag_y), id, font=font)
            draw.rectangle(text_box, fill=color_hex)
            draw.text((x1, tag_y), id, fill="white", font=font)

        except Exception as e:
            log.warning(f"Skipping malformed detection entry {det}: {e}")
            continue

    return annotated



def get_json_path_for_image(image_path: Path, layout_dir: Path) -> Path:
    """
    Resolves the JSON file path corresponding to a given image.

    Args:
        image_path:  Path to the source image.
        layout_dir:  Directory where JSON files are stored.

    Returns:
        Expected JSON path for the image.
    """
    return layout_dir / image_path.with_suffix(".json").name


def visualize(image_path: Path, layout_dir: Path, visual_dir: Path, label_colors: dict[str, str]) -> None:
    """
    Generates and saves a visualization for a single image.

    Checks if a visualization already exists — skips if so.
    Saves the output in the same format as the source image,
    mirroring the layout_dir structure under visual_dir.

    Args:
        image_path:   Path to the source image.
        layout_dir:   Directory containing JSON layout files.
        visual_dir:   Root directory to save visualization files.
        label_colors: Mapping of label -> hex color string.
    """

    try:
        json_path = get_json_path_for_image(image_path, layout_dir)
        formated_visual_path = get_visual_path(layout_dir, visual_dir)
        out_path = formated_visual_path / image_path.name

        #Skip if visualization already exists
        if out_path.exists():
            log.info(f"Visualization already exists, skippling: {out_path}")
            return
        
        #Check JSON exists
        if not json_path.exists():
            log.warning(f"No JSON found for '{image_path.name}' at {json_path.name} - skipping visualization.")
            return
        
        #Load detections
        with open(json_path) as f:
            data = json.load(f)

        detections = data.get("detections", [])
        if not detections:
            log.warning(f"No detections in {json_path.name} - skipping visualization.")
            return
        
        #Draw and save
        image = Image.open(image_path).convert("RGB")
        annotated = draw_bboxes_on_image(image, detections, label_colors)

        image_format = Image.open(image_path).format or "PNG"
        annotated.save(out_path, format=image_format)
        log.info(f"Saved visualizations: {out_path}")

    except json.JSONDecodeError as e:
        log.error(f"Corrupt JSON for {image_path.name}: {e}")

    except OSError as e:
        log.error(f"File I/O error while visualizing {image_path.name}: {e}")

    except Exception as e:
        log.exception(f"Unexpected error visualizing {image_path.name}: {e}")



if __name__ == "__main__":
    import sys

    log.info("=" * 50)
    log.info("Starting visualization tests...")
    log.info("=" * 50)

    TEST_IMAGE_PATH = Path("dataset/bank_statments/raw/images/10.jpg")
    TEST_LAYOUT_DIR = Path("tests/sample_data/json_outputs/bank_statments/outputs/json_of_images")
    TEST_VISUAL_DIR = Path("tests/visual_outputs/bank_statments/outputs/visual_of_images")
    LABEL_CONFIG    = Path("src/layout_detection_new/config/label_config.yaml")

    log.info("TEST 1: load_label_colors")
    label_colors = load_label_colors(LABEL_CONFIG)
    if label_colors:
        log.info(f"PASSED — loaded {len(label_colors)} label colors: {list(label_colors.keys())}")
    else:
        log.warning("FAILED — label colors empty, check your label_config.yaml path")
        sys.exit(1)

    log.info("TEST 2: get_json_path_for_image")
    TEST_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = get_json_path_for_image(TEST_IMAGE_PATH, TEST_LAYOUT_DIR)
    expected_json = TEST_LAYOUT_DIR / "10.json"
    if json_path == expected_json:
        log.info(f"PASSED — resolved JSON path: {json_path}")
    else:
        log.error(f"FAILED — expected {expected_json}, got {json_path}")
        sys.exit(1)

    log.info("TEST 3: get_visual_path")
    resolved_visual_dir = get_visual_path(TEST_LAYOUT_DIR, TEST_VISUAL_DIR)
    if resolved_visual_dir.exists():
        log.info(f"PASSED — mirrored visual directory created at: {resolved_visual_dir}")
    else:
        log.error(f"FAILED — visual directory was not created at: {resolved_visual_dir}")
        sys.exit(1)

    log.info("TEST 4: dummy JSON creation")
    dummy_detections = {
        "image_name": TEST_IMAGE_PATH.name,
        "detections": [
            {"id": 1, "label": "Title",   "bbox": [50,  30,  400, 80]},
            {"id": 2, "label": "Text",    "bbox": [50,  100, 400, 300]},
            {"id": 3, "label": "Table",   "bbox": [50,  320, 400, 600]},
            {"id": 4, "label": "Caption", "bbox": [50,  610, 400, 650]},
            {"id": 5, "label": "Picture", "bbox": [50,  660, 400, 900]},
        ]
    }
    with open(json_path, "w") as f:
        json.dump(dummy_detections, f, indent=4)
    if json_path.exists():
        log.info(f"PASSED — dummy JSON written to: {json_path}")
    else:
        log.error("FAILED — dummy JSON was not created")
        sys.exit(1)

    log.info("TEST 5: draw_bboxes_on_image")
    try:
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        annotated = draw_bboxes_on_image(image, dummy_detections["detections"], label_colors)
        if annotated.size == image.size:
            log.info(f"PASSED — annotated image size matches original: {annotated.size}")
        else:
            log.error(f"FAILED — size mismatch: original {image.size}, annotated {annotated.size}")
            sys.exit(1)
    except Exception as e:
        log.error(f"FAILED — draw_bboxes_on_image raised an exception: {e}")
        sys.exit(1)

    log.info("TEST 6: visualize — end-to-end")
    visualize(TEST_IMAGE_PATH, TEST_LAYOUT_DIR, TEST_VISUAL_DIR, label_colors)
    out_path = resolved_visual_dir / TEST_IMAGE_PATH.name   # ← uses mirrored dir
    if out_path.exists():
        log.info(f"PASSED — visualization saved at: {out_path}")
    else:
        log.error(f"FAILED — output file was not created at: {out_path}")
        sys.exit(1)

    # ── Test 7: skip if already exists ──────────────────────────────────────
    log.info("TEST 7: visualize — skip if visualization already exists")
    visualize(TEST_IMAGE_PATH, TEST_LAYOUT_DIR, TEST_VISUAL_DIR, label_colors)
    log.info("PASSED — re-run completed without crash (check logs for 'already exists')")

    log.info("=" * 50)
    log.info("All tests passed!")
    log.info("=" * 50)