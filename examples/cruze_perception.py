import os
import sys
import time
import requests
from PIL import Image
import torch

# Ensure the parent directory is in the path so we can import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.engine import VisionEngine

def print_telemetry_header():
    banner = """
    ========================================================================
     🌐  CRUZE AUTONOMOUS LOGISTICS | PERCEPTION TELEMETRY STREAM
     🚀  POWERED BY AETHERVISION LOCAL VISION ENGINE (MPS ACCELERATED)
    ========================================================================
    """
    print(banner)

def main():
    print_telemetry_header()
    
    # 1. Download a test road/traffic image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    print(f"[Cruze Telemetry] Downloading road test frame from Hugging Face...")
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw).convert("RGB")
        print(f"[Cruze Telemetry] Frame downloaded successfully: {img.width}x{img.height} pixels.")
    except Exception as e:
        print(f"[Error] Failed to download test image: {e}")
        return

    # 2. Instantiate local GPU-accelerated Vision Engine
    print("\n[Cruze Telemetry] Initializing local Perception Core...")
    engine = VisionEngine()
    
    # 3. Modality 1: Object Detection (Florence-2)
    print("\n[Cruze Telemetry] MODALITY 1/3: RUNNING Florence-2 VEHICLE DETECTION...")
    print("[Cruze Telemetry] Task: <OD> (Object Detection)")
    
    start_time = time.time()
    od_res = engine.run_florence_task(img, "<OD>")
    latency_od = time.time() - start_time
    
    bboxes = od_res["result"]["<OD>"]["bboxes"]
    labels = od_res["result"]["<OD>"]["labels"]
    
    print("\n------------------ DETECTED ROAD OBJECTS ------------------")
    for idx, (box, label) in enumerate(zip(bboxes, labels)):
        print(f" 🚘 [ID {idx:02d}] Type: {label:<10} | Coordinates (px): x=[{box[0]}, {box[2]}], y=[{box[1]}, {box[3]}]")
    print(f"----------------------------------------------------------")
    print(f"Perception latency (Florence-2): {latency_od:.2f} seconds.")

    # 4. Modality 2: Proximity & Depth Estimation (Depth Anything V2)
    print("\n[Cruze Telemetry] MODALITY 2/3: RUNNING Depth Anything V2 RANGE FINDER...")
    start_time = time.time()
    depth_res = engine.run_depth_map(img)
    latency_depth = time.time() - start_time
    
    stats = depth_res["stats"]
    print("\n------------------ PROXIMITY TELEMETRY ------------------")
    print(f" 📏 Mean Proximity Distance Indicator : {stats['mean']:.2f}")
    print(f" 🎚️ Relative Proximity Range           : Min={stats['min']:.2f} (Far), Max={stats['max']:.2f} (Close)")
    
    # Simulate autonomous vehicle collision warning
    if stats['max'] > 15.0:
         print(" ⚠️  [WARNING] OBSTACLE PROXIMITY IN CRITICAL RANGE - COLLISION RISK")
    else:
         print(" ✅ [SAFE] Obstacle clearance in safe driving envelope")
    print(f"---------------------------------------------------------")
    print(f"Range finder latency (Depth Anything V2): {latency_depth:.2f} seconds.")

    # 5. Modality 3: Scenario & Hazard Classification (CLIP)
    print("\n[Cruze Telemetry] MODALITY 3/3: RUNNING CLIP SCENARIO CLASSIFICATION...")
    road_states = [
        "a clear open highway",
        "heavy highway traffic congestion",
        "an emergency vehicle with flashing lights",
        "a delivery truck loading cargo",
        "a pedestrian crossing the road",
        "poor visibility night driving"
    ]
    
    start_time = time.time()
    clip_res = engine.run_similarity(img, road_states)
    latency_clip = time.time() - start_time
    
    print("\n------------------ SCENARIO MATCHING ------------------")
    for idx, match in enumerate(clip_res["matches"][:3]):
        prob_percent = match['probability'] * 100
        icon = "🎯" if idx == 0 else "🔹"
        print(f" {icon} Match #{idx+1}: {match['label']:<40} | Confidence: {prob_percent:.2f}%")
    print(f"-------------------------------------------------------")
    print(f"Scene classification latency (CLIP): {latency_clip:.2f} seconds.")
    
    print(f"\n[Cruze Telemetry] Total Pipeline Latency: {latency_od + latency_depth + latency_clip:.2f}s")
    print("[Cruze Telemetry] Live perception stream complete. Status: ONLINE")

if __name__ == "__main__":
    main()
