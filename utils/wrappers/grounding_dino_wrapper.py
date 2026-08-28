import os
import torch
import cv2
import numpy as np
from PIL import Image

class GroundingDinoPredictor:
    """
    A lightweight wrapper around HuggingFace Transformers' Grounding DINO.
    It mimics the signature of a Detectron2 predictor but requires a text prompt.
    """
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", box_threshold=0.25, text_threshold=0.25):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
            
        print(f">>> [GroundingDinoWrapper] Initializing model {model_id}...")
        
        self.device = "cpu"
        
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.grounding_dino")
            
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
    def __call__(self, image_np):
        print("Warning: GroundingDinoPredictor requires a text prompt. Calling .predict() with default prompt.")
        return self.predict(image_np, "part. object.")
        
    def predict(self, image_np, text_prompt):
        if isinstance(image_np, np.ndarray):
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_np
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image_np

        if not text_prompt.endswith("."):
            text_prompt += "."

        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[pil_image.size[::-1]]
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil_image.size[::-1]]
            )[0]
            
        detections = []
        labels_list = results.get("text_labels", results.get("labels", []))
        
        for score, label, box in zip(results["scores"], labels_list, results["boxes"]):
            score_val = score.item()
            if score_val < self.box_threshold:
                continue
                
            box_coords = [float(x) for x in box.tolist()]
            
            detections.append({
                'class_name': label,
                'bbox': box_coords,
                'score': score_val
            })
            
        return detections

def load_grounding_dino_model(model_id="IDEA-Research/grounding-dino-tiny", box_threshold=0.25):
    return GroundingDinoPredictor(model_id=model_id, box_threshold=box_threshold)
