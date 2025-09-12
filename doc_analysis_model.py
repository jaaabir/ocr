# Install ultralytics if not already installed
# pip install ultralytics

from ultralytics import YOLO
from PIL import Image
import torch

def load_doclaynet_model(model_name="malaysia-ai/YOLOv8X-DocLayNet-Full-1024-42", device=None):
    """
    Load YOLOv8X model pretrained on DocLayNet for layout detection.

    Args:
        model_name (str): Hugging Face YOLOv8X model name
        device (str or None): 'cuda', 'mps', or 'cpu'. If None, auto-detect GPU if available.

    Returns:
        model: YOLO model ready for inference
        device: torch device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = YOLO(model_name)
    model.to(device)
    
    return model, device


def get_doc_bboxes(model, images, conf_thresh=0.5, imgsz=842):
    """
    Run the model on one or multiple images and extract bounding boxes + class labels.

    Args:
        model: YOLOv8X-DocLayNet model
        images (np.array or list of np.array): Input image(s)
        conf_thresh (float): Minimum confidence threshold for detections

    Returns:
        List of lists: Each sublist corresponds to detections for one image:
        [[{'label': str, 'score': float, 'bbox': [x_min, y_min, x_max, y_max]}, ...], ...]
    """
    results = model(images, imgsz=imgsz, conf=conf_thresh)
    all_outputs = []

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy().tolist()
        all_outputs.append(boxes)

    return all_outputs


