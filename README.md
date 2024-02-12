# Video Object Detection with DETR and Gaudi2

This Python script utilizes the DETR (DEtection TRansformer) model for object detection in videos. It processes a video file, detects objects frame by frame, and outputs a new video with bounding boxes drawn around the detected objects. The script is designed to run on systems with Gaudi2 support, leveraging the power of the Habana Gaudi2 AI processors for efficient deep learning inference.

## Prerequisites

- Python 3.6 or later
- OpenCV
- PyTorch
- Transformers library
- PIL (Python Imaging Library)

Ensure that your environment is set up to support Habana Gaudi2 processors if you intend to leverage them for running this script.

## Installation

1. Clone this repository or download the script directly.
2. Install the required Python packages:

```bash
pip install torch torchvision opencv-python pillow transformers
```

3. Ensure that your system is configured correctly for Habana Gaudi2 processors. Follow the official [Habana documentation](https://docs.habana.ai/) for setup instructions.

## Usage

1. Modify the `video-object-detection-gaudi2.py` script to point to the input video file by changing the `cv2.VideoCapture` path:

```python
cap = cv2.VideoCapture('/path/to/your/video/file.mov')
```

2. Adjust the output path and video settings in the `cv2.VideoWriter` function as necessary:

```python
out = cv2.VideoWriter('/path/to/output/video.avi', fourcc, 20.0, (width, height))
```

3. Run the script:

```bash
python video-object-detection-gaudi2.py
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
