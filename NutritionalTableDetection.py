from ultralytics import YOLO
from PIL import Image
import uuid
import os

model = YOLO("model/yolov10.pt")


def highlight_boxes(img):
    result = model.predict(img)[0]
    if len(result) == 0:
        return None, None

    image_uuid = uuid.uuid4().hex
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{image_uuid}.jpg"
    result.save(temp_path)

    with Image.open(temp_path) as image:
        max_size = 800
        image.thumbnail((max_size, max_size))
        image.save(temp_path, "JPEG", quality=85)

    print(result.boxes.xyxyn)

    return image_uuid, result.boxes.xyxyn.tolist()

