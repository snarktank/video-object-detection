# Video Object Detection with DETR and Gaudi2

This Python script utilizes the [DETR (DEtection TRansformer) model](https://huggingface.co/facebook/detr-resnet-50) for object detection in videos. It processes a video file, detects objects frame by frame, and outputs a new video with bounding boxes drawn around the detected objects. The script is designed to run on systems with Gaudi2 support, leveraging the power of the Habana Gaudi2 AI processors for efficient deep learning inference.

You can provision a Gaudi2® Deep Learning Server (8 x Gaudi2® HL-225H mezzanine cards - 3rd Gen Xeon® 8380 Processor) @ $10.42 / hour with 1 TB RAM and 30 TB disk by creating an account at [https://console.cloud.intel.com/](https://console.cloud.intel.com/).

## Prerequisites

- Python 3.6 or later
- OpenCV
- PyTorch
- Transformers library
- PIL (Python Imaging Library)

## Installation

1. Clone this repository or download the script directly.
2. Install the required Python packages:

```bash
pip install torch torchvision opencv-python pillow transformers
```

## Usage

1. Modify the `process.py` script to point to the input video file by changing the `cv2.VideoCapture` path:

```python
cap = cv2.VideoCapture('/path/to/your/video/file.mov')
```

2. Adjust the output path and video settings in the `cv2.VideoWriter` function as necessary:

```python
out = cv2.VideoWriter('/path/to/output/video.avi', fourcc, 20.0, (width, height))
```

3. Run the script:

```bash
python process.py
```

The script will process the input video, detect objects in each frame, and output a new video with bounding boxes and labels for detected objects.

## Features

- Uses the DETR model for object detection, leveraging its transformer-based architecture for accurate and efficient detection.
- Processes video files frame by frame, drawing bounding boxes around detected objects.
- Outputs a new video with detected objects highlighted.
- Designed to utilize Habana Gaudi2 processors for accelerated inference, though it can run on any system with PyTorch support.

## Limitations

- The performance and speed of object detection depend on the system's hardware capabilities.
- Currently, the script is set up to process a single video file at a time. Batch processing is not supported.

## Contributing

Contributions to improve the script or extend its capabilities are welcome. Please submit a pull request or open an issue to discuss your ideas.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
