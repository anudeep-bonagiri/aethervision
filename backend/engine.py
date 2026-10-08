import base64
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForDepthEstimation,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)


class VisionEngine:
    def __init__(self):
        # Determine the best hardware device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[AetherVision] Apple Silicon GPU (MPS) is available and will be used.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[AetherVision] CUDA GPU is available and will be used.")
        else:
            self.device = torch.device("cpu")
            print("[AetherVision] GPU not found. Falling back to CPU.")
            
        # Lazy model loading holders
        self._florence_model = None
        self._florence_processor = None
        
        self._depth_model = None
        self._depth_processor = None
        
        self._clip_model = None
        self._clip_processor = None

    def load_florence(self):
        if self._florence_model is None:
            print("[AetherVision] Loading Florence-2 model (microsoft/Florence-2-base)...")
            start = time.time()
            # Note: trust_remote_code=True is required for custom model architectures
            self._florence_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", 
                trust_remote_code=True
            )
            # Florence-2 works best with float16 or float32. MPS handles float32 very reliably.
            self._florence_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base", 
                trust_remote_code=True
            ).to(self.device).eval()
            print(f"[AetherVision] Florence-2 loaded in {time.time() - start:.2f} seconds.")
        return self._florence_model, self._florence_processor

    def load_depth(self):
        if self._depth_model is None:
            print("[AetherVision] Loading Depth Anything V2 (depth-anything/Depth-Anything-V2-Small-hf)...")
            start = time.time()
            self._depth_processor = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            self._depth_model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            ).to(self.device).eval()
            print(f"[AetherVision] Depth Anything loaded in {time.time() - start:.2f} seconds.")
        return self._depth_model, self._depth_processor

    def load_clip(self):
        if self._clip_model is None:
            print("[AetherVision] Loading CLIP model (openai/clip-vit-base-patch32)...")
            start = time.time()
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
            print(f"[AetherVision] CLIP loaded in {time.time() - start:.2f} seconds.")
        return self._clip_model, self._clip_processor

    def run_florence_task(self, image: Image.Image, task_prompt: str, text_input: str | None = None) -> dict:
        """
        Runs a Florence-2 task on the input image.
        Common tasks: 
          - '<CAPTION>': Simple caption
          - '<DETAILED_CAPTION>': Detailed caption
          - '<MORE_DETAILED_CAPTION>': More detailed description
          - '<OD>': Object Detection (returns bounding boxes)
          - '<OCR>': Optical Character Recognition
          - '<OCR_WITH_REGION>': OCR with pixel bounding boxes
          - '<DENSE_REGION_CAPTION>': Predicts dense boxes + labels
          - '<REFERRING_EXPRESSION_SEGMENTATION>': Custom expression segmenting (needs text_input)
        """
        model, processor = self.load_florence()
        
        # Prepare inputs
        prompt = task_prompt
        if text_input and task_prompt == "<REFERRING_EXPRESSION_SEGMENTATION>":
            prompt = f"{task_prompt}{text_input}"
            
        print(f"[AetherVision] Running Florence-2 task: {prompt}")
        start_time = time.time()
        
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        latency = time.time() - start_time
        print(f"[AetherVision] Florence-2 execution finished in {latency:.2f}s.")
        
        return {
            "result": parsed_answer,
            "latency": latency,
            "raw_output": generated_text
        }

    def run_depth_map(self, image: Image.Image) -> dict:
        """
        Computes the depth map of the input image using Depth-Anything-V2.
        Returns a base64 encoded colorized depth map and raw depth metrics.
        """
        model, processor = self.load_depth()
        
        print("[AetherVision] Running Depth Anything V2...")
        start_time = time.time()
        
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        post_processed = processor.post_process_depth_estimation(
            outputs, 
            target_sizes=[(image.height, image.width)]
        )
        
        # Convert depth tensor to numpy array
        depth_map = post_processed[0]["predicted_depth"].cpu().numpy()
        
        # Normalize depth map to 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min + 1e-8) * 255.0).astype(np.uint8)
        
        # Colorize depth map (using INFERNO colormap for stunning visual presentation)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        # Convert to Base64 to send to UI
        _, buffer = cv2.imencode('.png', depth_color)
        depth_base64 = base64.b64encode(buffer).decode("utf-8")
        
        latency = time.time() - start_time
        print(f"[AetherVision] Depth Anything execution finished in {latency:.2f}s.")
        
        return {
            "depth_image": f"data:image/png;base64,{depth_base64}",
            "latency": latency,
            "stats": {
                "min": float(depth_min),
                "max": float(depth_max),
                "mean": float(depth_map.mean())
            }
        }

    def run_similarity(self, image: Image.Image, text_candidates: list) -> dict:
        """
        Runs CLIP classification to find similarity matches between the uploaded image
        and a list of user-provided text labels.
        """
        model, processor = self.load_clip()
        
        print(f"[AetherVision] Running CLIP classification against {len(text_candidates)} classes...")
        start_time = time.time()
        
        inputs = processor(
            text=text_candidates, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get softmax probabilities
        logits_per_image = outputs.logits_per_image # image-to-text similarity logits
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        results = [
            {"label": label, "probability": float(prob)} 
            for label, prob in zip(text_candidates, probs)
        ]
        # Sort by probability descending
        results.sort(key=lambda x: x["probability"], reverse=True)
        
        latency = time.time() - start_time
        print(f"[AetherVision] CLIP execution finished in {latency:.2f}s.")
        
        return {
            "matches": results,
            "latency": latency
        }
