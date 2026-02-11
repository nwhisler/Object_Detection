# Object Detection (Real‑Time Webcam Inference)

## Overview

This project is a minimal real‑time object detection demo that:

* Captures frames from a webcam (via OpenCV)
* Runs inference using a **pretrained RetinaNet (ResNet‑50) model** from **KerasCV**
* Draws **bounding boxes + class labels** on the live video stream

The entry point is:

* `ObjectDetection.py`

---

## How it works

`ObjectDetection.py`:

1. Opens a camera device with `cv2.VideoCapture(<index>)`
2. Downloads/loads a pretrained KerasCV model preset:

   * `keras_cv.models.RetinaNet.from_preset("retinanet_resnet50_pascalvoc", bounding_box_format="xywh")`
3. Resizes frames to `640×640` (padding to aspect ratio) for inference
4. Predicts bounding boxes/classes per frame and visualizes them
5. Displays the annotated stream in an OpenCV window
6. Exits when you press **`q`**

The model is trained on **Pascal VOC**, so detections are limited to the VOC label set.

---

## Requirements

### System

* Python 3.9+ recommended
* A working webcam

### Python packages

The script imports:

* `tensorflow`
* `keras-cv`
* `opencv-python` (`cv2`)
* `numpy`
* `matplotlib`
* `tqdm`

You can install them with:

```bash
pip install tensorflow keras-cv opencv-python numpy matplotlib tqdm
```

---

## Run

From the directory containing `ObjectDetection.py`:

```bash
python ObjectDetection.py
```

A window titled `image` will open showing the webcam feed with bounding boxes.

### Quit

Press **`q`** to exit cleanly.

---

## Camera selection

The script currently uses:

```python
cap = cv2.VideoCapture(1)
```

* Use `0` for most built-in cameras.
* Use `1`, `2`, … for external USB cameras or additional devices.

If you see a black window or `cap.read()` fails, change the index.

---

## Output labels

The script provides a Pascal VOC class mapping:

* Aeroplane, Bicycle, Bird, Boat, Bottle,
  Bus, Car, Cat, Chair, Cow,
  Dining Table, Dog, Horse, Motorbike, Person,
  Potted Plant, Sheep, Sofa, Train, TV/Monitor

The visualization uses `class_mapping=id2label` so predicted class IDs render as readable labels.

---

## Troubleshooting

### Window opens but no image / crashes

* Verify the camera index (try `0`).
* Ensure no other program is using the camera.

### Slow FPS

* Use a GPU-enabled TensorFlow build if available.
* Reduce input resolution (e.g., resize to `512×512`) to speed up inference.

### Import errors

* Confirm you installed the packages into the same environment you run the script from.

---
