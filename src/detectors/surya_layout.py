import os
import json
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

DATA_DIR = "dataset_mtd_360/Multilingual Table Detection (MTD-360)/data"

def initialize_surya_layout_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The device the model is loaded to is: {device}")
    foundation_predictor = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    layout_predictor = LayoutPredictor(foundation_predictor)
    foundation_predictor.model.to(device)
    return foundation_predictor, layout_predictor


def table_detection_for_one_image_with_surya(image_path):
    OUT_DIR = "predictions_visualized"
    os.makedirs(OUT_DIR, exist_ok=True)

    foundation_predictor, layout_predictor = initialize_surya_layout_model()

    IMAGE_PATH = os.path.join(DATA_DIR, image_path)

    image = Image.open(IMAGE_PATH)
    draw = ImageDraw.Draw(image) #To draw on the image

    layout_predictions = layout_predictor([image])

    for layoutResult in layout_predictions:
        for bbox in layoutResult.bboxes:

            if bbox.label=='Table':
                coords=bbox.bbox
                draw.rectangle(coords, outline="green", width=2)
                draw.text((coords[0], coords[1]-15), fill="green", text="Detected")

    save_path = "detection_for_one_image"+image_path
    image.save(save_path)


def table_detection_for_batch_images_with_surya(batch_size=8, pred_file="predictions.json"):
    OUT_DIR = "predictions_visualized"
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(pred_file):
        print(f"Predictions already exist")
        with open(pred_file) as file:
            pred_data = json.load(file)

        return pred_data

    pred_data = {}

    foundation_predictor, layout_predictor = initialize_surya_layout_model()

    IMAGE_PATHS = [os.path.join(DATA_DIR,image) for image in os.listdir(DATA_DIR) if image.lower().endswith(('.jpg', '.png', '.jepg'))]

    print(f"Initiating the table detection for {len(IMAGE_PATHS)} images =")

    for i in tqdm(range(0,len(IMAGE_PATHS), batch_size)):
        batch_paths = IMAGE_PATHS[i: i+batch_size]
        batch_images = [Image.open(EACH_IMAGE_PATH) for EACH_IMAGE_PATH in batch_paths]
        batch_results = layout_predictor(batch_images)

        for img, result, path in zip(batch_images, batch_results, batch_paths):

            draw = ImageDraw.Draw(img)
            pred_bboxes = []

            for bbox in result.bboxes:
                if bbox.label=='Table':
                    coords=bbox.bbox
                    pred_bboxes.append(coords)
                    draw.rectangle(coords, outline="green", width=2)
                    draw.text((coords[0], coords[1]-15), fill="green", text="Detected")

            filename = os.path.basename(path)
            pred_data[filename] = pred_bboxes
            save_path = os.path.join(OUT_DIR,filename)
            img.save(save_path)

    with open(pred_file, "w") as f:
        json.dump(pred_data, f)

    print(f"Saved predictions of all the {len(IMAGE_PATHS)} images")

    return pred_data


if __name__ == "__main__":
    IMAGE_PATH = "Annualreport2017-18-Hin-Eng_page-0174_jpg.rf.68db473edfdf713b899ea4c5ba9b46a3.jpg"
    pred_data = table_detection_for_batch_images_with_surya()
    print(f"Prediction data is: {pred_data[IMAGE_PATH]}")


# IMAGE_PATH = "dataset_mtd_360/Multilingual Table Detection (MTD-360)/data/Annualreport2017-18-Hin-Eng_page-0175_jpg.rf.70a62874d13ab5a65d80e765105596e8.jpg"
# table_recognizer = TableRecognition()

#Initialize models


# image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# print(f"Found {len(image_files)} number of files to detect tables")

# for filename in tqdm(image_files):
#     full_path = os.path.join(DATA_DIR,filename)

#     image = Image.open(full_path)
#     draw = ImageDraw.Draw(image)

#     #Run layout predictions
#     layout_predictions = layout_predictor([image])

#     for layoutResult in layout_predictions:
#         for bbox in layoutResult.bboxes:

#             if bbox.label=='Table':
#                 coords=bbox.bbox
#                 draw.rectangle(coords, outline="green", width='2')
#                 draw.text((coords[0], coords[1]-15), fill="green")

#     save_path = os.path.join(OUT_DIR,filename)
#     image.save(save_path)


# # padding = 10
# # image = Image.open(IMAGE_PATH)
# # draw = ImageDraw.Draw(image)
# # layout_predictions = batch_layout_detection([image], model, processor)
# # layout_predictions = layout_predictor([image])

# # for layoutResult in layout_predictions:
# #     for bbox in layoutResult.bboxes:

# #         if bbox.label=='Table':
# #             coords=bbox.bbox
# #             # print(f"The coordinates are: {coords[0]-padding}")
# #             draw.rectangle(coords, outline="green", width=5)
# #             draw.text((coords[0], coords[1] - 15), "DETECTED TABLE", fill="green")

#             # #Coordinates to crop the tables from the image, but before that we add padding, so that the lines of the table are not edited out
#             # padded_coords = (
#             #     max(0,coords[0]-padding),
#             #     max(0,coords[1]-padding),
#             #     min(image.width,coords[2]+padding),
#             #     min(image.height,coords[3]+padding)
#             # )

#             # table_img = image.crop(padded_coords)
#             # table_img.convert("RGB").save("croped_table_1.jpg")

#             # table_result = table_recognizer.recognition(table_img)

#             # print(table_result)


# # image.convert("RGB").save("table_detection_pred_visualize.png")
# # image.show()