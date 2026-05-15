# Visual Product Search — Streamlit Demo 🛍️

A beautiful, interactive web interface for the **Visual Product Search** pipeline. This application allows users to upload any clothing image, automatically crop the garment using YOLO, and instantly search through the DeepFashion gallery using our custom-trained CLIP and BLIP-ITM models.

## ✨ Features
* **Interactive Garment Detection**: Upload an image and let YOLO detect the upper, lower, or full-body garments. If multiple items are found, you can visually select which one to search for!
* **Full Pipeline Execution**: Runs the complete evaluation pipeline (CLIP embedding → HNSW Nearest Neighbor Search → BLIP-ITM Cross-Attention Reranking) in real-time.
* **Dynamic Configuration**: Use the sidebar to seamlessly switch between different model ablations (Baseline A, Vision+Text B, Best C), text-fusion weights ($\alpha$), and training seeds.
* **Beautiful UI**: Custom-styled CSS with similarity progress bars, ranking badges, and rich image cards.

## 🛠️ Prerequisites & Installation

Make sure you have installed the required libraries. If you haven't already, install them via pip:

```bash
pip install streamlit torch torchvision Pillow ultralytics open_clip_torch transformers hnswlib pandas numpy
```

## 📂 Directory Layout

Before running the application, ensure your models and pre-computed HNSW indexes are placed correctly. The application expects the YOLO weights (`best.pt`) in the root folder, and all other checkpoints in a `checkpoints2/` directory:

```text
Demo Application/
├── app.py
├── best.pt                       <-- Fine-tuned YOLO object detection weights
└── checkpoints2/                 <-- All other models and indexes
    ├── captions_cache.json
    ├── clip_seed14.pt            <-- Seed 14 CLIP model
    ├── clip_seed37.pt            <-- Seed 37 CLIP model
    ├── clip_seed106.pt           <-- Seed 106 CLIP model
    ├── meta_A_a1.0.json          <-- Metadata for index
    ├── hnsw_A_a1.0.index         <-- HNSW pre-computed index
    └── ... (other indexes and metadata files)
```

## 🚀 Running the App

Start the Streamlit development server by running:

```bash
streamlit run app.py
```

This will automatically open the web application in your default browser (usually at `http://localhost:8501`).

### Sidebar Configuration
Once the app is running, use the **Sidebar** to configure the application. 

> [!IMPORTANT]
> **Dataset Image Root is Required**
> In order for the application to actually display the gallery images in the search results, you **must** provide the absolute path to your dataset in the sidebar's "Dataset image root" field (e.g., `/path/to/deepfashion/img`). If you leave this blank, the app will still work, but you will only see blank placeholder boxes instead of the retrieved clothing images!

Additionally, you can:
1. Select your Model Configuration (e.g., `C α=0.7 (best)`).
2. Change the Checkpoint Seed.
3. Choose how many Top-K results you want to retrieve.
