# AetherVision 🪐

AetherVision is a local, high-performance, open-source computer vision agent workbench and library. It is designed to run state-of-the-art vision models locally on consumer hardware, with dedicated support for **Apple Silicon GPU (Metal Performance Shaders / MPS)** acceleration.

AetherVision exposes a unified vision inference server and features a premium, interactive glassmorphic web dashboard to visually explore object detection, segmentations, dense region descriptions, optical character recognition (OCR), depth estimation, and CLIP similarity comparison.

---

## ⚡ Core Features & Models

AetherVision integrates three leading lightweight open-source vision models optimized for fast inference and low memory consumption:

1. **Microsoft Florence-2-base** (`microsoft/Florence-2-base` - 232M Parameters)
   - Handles multi-task vision prompts: image captioning, object detection (`<OD>`), dense region captioning, OCR text extraction, and referring expression segmentation.
2. **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf` - 25M Parameters)
   - Estimates pixel-level depth maps with stunning colorization mapping (INFERNO colormap).
3. **OpenAI CLIP** (`openai/clip-vit-base-patch32` - 150M Parameters)
   - Computes contrastive image-text similarity embeddings for zero-shot categorization.

---

## 🏗️ Architecture

```
aethervision/
├── backend/
│   ├── app.py           # FastAPI server & route handlers
│   └── engine.py        # Model loader and inference engine (Florence-2, Depth V2, CLIP)
├── frontend/
│   ├── index.html       # Workbench UI structure
│   ├── style.css        # Premium dark glassmorphism stylesheet
│   └── app.js           # Canvas annotation & API communications handler
├── requirements.txt     # Python package dependencies
├── README.md            # Documentation
└── LICENSE              # MIT License
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- macOS (Apple Silicon M1/M2/M3/M4) or any system with CUDA/CPU support.

### 1. Clone & Set Up Directory
Ensure you are in the project folder:
```bash
cd aethervision
```

### 2. Set Up Virtual Environment & Dependencies
We recommend using `uv` (a fast Python package installer) or `pip`:
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
uv pip install -r requirements.txt
```

### 3. Run the Server
Launch the local FastAPI application server:
```bash
python -m backend.app
```
The server will start and automatically serve the static web dashboard on:
**[http://localhost:8000](http://localhost:8000)**

---

## 🖥️ Usage Guide

1. **Upload an Image**: Drag-and-drop or click the viewport upload box to select an image.
2. **Select Modality**:
   - **Florence-2**: Select a sub-task (e.g., *Object Detection* or *OCR*). If *Referring Expression Segmentation* is selected, provide a target description (e.g., "the cup on the table").
   - **Depth Estimation**: Computes a dense depth map. Brighter violet/orange regions represent objects closer to the camera.
   - **CLIP Classification**: Input comma-separated text descriptors (e.g., "a dog, a cat, a car") to calculate image-text alignment probabilities.
3. **Run Inference**: Press the **Run Inference** button to execute the pipeline locally on your Apple Silicon GPU.
4. **Inspect Console**: Switch between the **Analysis Result** tab (visual outputs list) and the **Raw Response** tab (raw JSON output).

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
