# MyNutriApps AI Server

* ResNet18 CNN Classification: https://colab.research.google.com/drive/1qNTIy4Gbdm7_ceirV8keaeTqQ2GW7Ynv
* YOLOv10 Nutritional Table Detection: https://colab.research.google.com/drive/1wKVDwXzL_eGgjGcDFCQ0zCosQJzZJkGm

## Packages needed
* [torch<2.6.0](https://pytorch.org/)
* torchvision
* [transformers](https://huggingface.co/docs/transformers/en/installation)
* [ultralytics](https://docs.ultralytics.com/quickstart/)
* flask and Flask-APScheduler
* Production: waitress

For new environment, run `apt-get install -y libgl1-mesa-dev ibglib2.0-0`. See [issue](https://github.com/ultralytics/ultralytics/issues/1270).
