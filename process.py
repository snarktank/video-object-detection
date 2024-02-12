import cv2  # Import OpenCV library for computer vision tasks
import torch  # Import PyTorch library for deep learning tasks
from transformers import DetrForObjectDetection, DetrImageProcessor  # Import specific classes for object detection from the transformers library
from PIL import Image  # Import the Python Imaging Library (PIL) for image processing tasks

# Load the DETR (DEtection TRansformer) model and its processor for object detection
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')  # Load the model with pretrained weights
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")  # Load the processor for preparing inputs
model.eval()  # Set the model to evaluation mode

# Define a function to draw bounding boxes around detected objects
def draw_boxes(boxes, labels, scores, image):
    for box, label, score in zip(boxes, labels, scores):  # Iterate over detected objects
        if score > 0.5:  # Filter out detections with a score lower than 0.5
            box = box.astype(int)  # Convert box coordinates to integers
            # Draw a rectangle around the detected object
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Put a label with the object class and score above the rectangle
            cv2.putText(image, f'{label}: {score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Change this line to use the video file instead of a camera
cap = cv2.VideoCapture('./your_movie.mov')  # Adjust the path as necessary

# Define the codec and create VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./your_movie_processed.avi', fourcc, 20.0, (1620, 1080))  # Adjust the frame size to match your_movie.mov

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
processed_frames = 0  # Initialize a counter for processed frames

if not cap.isOpened():  # Check if the video was successfully opened
    print("Error: Could not open video.")
else:
    print(f"Success: Video opened. Total frames: {total_frames}")
    while True:  # Start an infinite loop to process video frames
        print(f"Reading a new frame... ({processed_frames + 1}/{total_frames})")
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:  # If frame is not read correctly
            print("Failed to grab frame")
            break  # Exit the loop

        print("Converting frame to PIL Image format...")
        # Convert the captured frame to PIL Image format
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        print("Preparing frame for the model...")
        # Prepare the frame for the model using the processor
        inputs = processor(images=pil_image, return_tensors="pt")

        print("Making predictions on the frame...")
        # Make predictions on the frame with the model
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)

        print("Processing model outputs...")
        # Process the model's outputs to extract bounding boxes, labels, and scores
        target_sizes = torch.tensor([pil_image.size[::-1]])  # Get the size of the input image
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        print("Extracting boxes, labels, and scores from the results...")
        # Extract boxes, labels, and scores from the results
        boxes = results['boxes'].cpu().numpy()  # Convert boxes to numpy array
        labels = [model.config.id2label[label.item()] for label in results['labels']]  # Convert label indices to string labels
        scores = results['scores'].cpu().numpy()  # Convert scores to numpy array

        print("Drawing bounding boxes on the original frame...")
        # Draw bounding boxes on the original frame
        draw_boxes(boxes, labels, scores, frame)

        print("Writing the processed frame to the output video...")
        # Write the frame with detected objects to the output video
        out.write(frame)

        processed_frames += 1  # Increment the processed frames counter

    print(f"Finished processing. Total processed frames: {processed_frames}/{total_frames}")

    # Release the video capture, video writer, and close all OpenCV windows when the loop is exited
    cap.release()
    out.release()
    cv2.destroyAllWindows()
