import torch
import torchvision
import cv2
import numpy as np

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')  # Ensure weights are properly loaded
model.eval()  # Set the model to evaluation mode

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Define a function to load and transform the image
def load_image(image_path, resize_dims=(800, 600)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_resized = cv2.resize(image_rgb, resize_dims)  # Resize the image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image_resized).unsqueeze(0), image_resized  # Return resized image for visualization

# Define a function to perform object detection
def detect_objects(image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

# Define a function to visualize the results using OpenCV
def visualize_results(original_image, predictions, threshold=0.5):
    for i, (box, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['scores'])):
        if score > threshold:
            xmin, ymin, xmax, ymax = box.numpy().astype(int)
            cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label_id = int(predictions[0]["labels"][i])
            # Check if label_id is within the bounds of COCO_CLASSES
            if label_id < len(COCO_CLASSES):
                label_name = COCO_CLASSES[label_id]  # Get the class name
            else:
                print(f"Warning: label_id {label_id} is out of range.")
                label_name = "Unknown"
            label = f'{label_name}: {score:.2f}'  # Use class name instead of ID
            cv2.putText(original_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Object Detection", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Provide the path to the image file
    image_path = r'C:/Users/phone/Downloads/image8.jpg'  # Change this to your image path

    # Load and process the image
    image_tensor, original_image = load_image(image_path)

    # Detect objects in the image
    predictions = detect_objects(image_tensor)

    # Visualize the results
    visualize_results(original_image, predictions)
