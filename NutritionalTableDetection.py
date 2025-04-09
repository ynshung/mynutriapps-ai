from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os
import uuid

model = YOLO("model/yolov10.pt")


def get_nutritional_boxes(img):
    result = model.predict(img)[0]
    boxes = result.tojson()
    return boxes


def highlight_boxes(img):
    result = model.predict(img)[0]
    image_uuid = uuid.uuid4().hex
    temp_path = f"temp/{image_uuid}.jpg"
    result.save(temp_path)

    with Image.open(temp_path) as img_resized:
        img_resized.thumbnail((800, 800))
        buffer = BytesIO()
        img_resized.save(buffer, format="JPEG")
        buffer.seek(0)

    os.remove(temp_path)
    return buffer


def get_cropped_image(img):
    result = model.predict(img)[0]
    if not result.boxes:
        return None
    image_uuid = uuid.uuid4().hex
    temp_path = f"temp/nutrition_table/{image_uuid}.jpg"
    result.save_crop("temp", file_name=f"{image_uuid}")

    with Image.open(temp_path) as img_resized:
        buffer = BytesIO()
        img_resized.save(buffer, format="JPEG")
        buffer.seek(0)

    os.remove(temp_path)
    return buffer
