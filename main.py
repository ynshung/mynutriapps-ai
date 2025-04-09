import FoodProductImageClassification as fpic
import NutritionalTableDetection as ntd
from flask_apscheduler import APScheduler
from flask import Flask, request, jsonify, send_file
from PIL import Image
import json
import os
from flask import send_from_directory

app = Flask(__name__)


@app.route("/status")
def home():
    return jsonify({"status": "ok"}), 200


@app.route("/api/v1/fpic", methods=["POST"])
def food_product_image_classification():
    if "image" not in request.files:
        return jsonify({"status": "error"}), 400

    image = request.files["image"]

    try:
        image = Image.open(image.stream)
        result = fpic.predict_image(image)
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        return jsonify({"status": str(e)}), 500


@app.route("/api/v1/ntd", methods=["POST"])
def nutritional_table_detection_preview():
    if "image" not in request.files:
        return jsonify({"status": "error"}), 400

    image = request.files["image"]

    try:
        image = Image.open(image.stream)
        boxes_uuid, result = ntd.highlight_boxes(image)
        if boxes_uuid is None:
            return (
                jsonify(
                    {"status": "no content", "message": "No nutritional table detected"}
                ),
                400,
            )
        return jsonify({"status": "success", "data": result, "uuid": })
    except Exception as e:
        return jsonify({"status": str(e)}), 500


@app.route("/boxes/<path:filename>")
def serve_boxes(filename):
    temp_folder = os.path.join(os.getcwd(), "temp")
    file_path = os.path.join(temp_folder, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=False)
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404


def clear_temp_folder():
    temp_folder = os.path.join(os.getcwd(), "temp")
    if os.path.exists(temp_folder):
        images = [
            os.path.join(temp_folder, f)
            for f in os.listdir(temp_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(images) > 128:
            images.sort(key=os.path.getctime)  # Sort by creation time
            for image in images[:64]:  # Remove the oldest 64 images
                os.remove(image)


scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()
scheduler.add_job(
    id="test-job", func=clear_temp_folder, trigger="interval", seconds=1000
)


def create_app():
    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5432)
