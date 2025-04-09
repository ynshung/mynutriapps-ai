import FoodProductImageClassification as fpic
import NutritionalTableDetection as ntd
from flask import Flask, request, jsonify, send_file
from PIL import Image
import json

app = Flask(__name__)


@app.route("/status")
def home():
    return jsonify({"status": "ok"}), 200


@app.route("/api/v1/fpic", methods=["POST"])
def food_product_image_classification():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    images = request.files.getlist("image")
    if not images:
        return jsonify({"error": "No image files provided"}), 400

    results = []
    for image in images:
        try:
            img = Image.open(image.stream)
            result = fpic.predict_image(img)
            results.append({"filename": image.filename, "result": result})
        except Exception as e:
            results.append({"filename": image.filename, "error": str(e)})

    return jsonify({"success": True, "data": results}), 200


def process_nutritional_table_request(processor_function, response_type="json"):
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]

    try:
        image = Image.open(image.stream)
        result = processor_function(image)
        if result is None:
            return jsonify({"error": "No nutritional table detected"}), 404

        if response_type == "json":
            result_obj = json.loads(result)[0]
            return jsonify({"success": True, "data": result_obj}), 200
        elif response_type == "image":
            return send_file(result, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/ntd/json", methods=["POST"])
def nutritional_table_detection_json():
    return process_nutritional_table_request(
        ntd.get_nutritional_boxes, response_type="json"
    )


@app.route("/api/v1/ntd/preview", methods=["POST"])
def nutritional_table_detection_preview():
    return process_nutritional_table_request(ntd.highlight_boxes, response_type="image")


@app.route("/api/v1/ntd/cropped", methods=["POST"])
def nutritional_table_detection_cropped():
    return process_nutritional_table_request(
        ntd.get_cropped_image, response_type="image"
    )


def create_app():
    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5432)
