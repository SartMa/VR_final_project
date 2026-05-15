# Batch Evaluation for Visual Product Search (`batch_eval.py`)

`batch_eval.py` is a batch evaluation script that computes the performance metrics of a Visual Product Search model. It computes **Recall@K**, **NDCG@K**, and **mAP@K** (for K = 5, 10, 15) by evaluating query images against a gallery of images. 

The script utilizes a combination of **YOLO** (for cropping/object detection), **CLIP** (for vision-language embeddings), and **BLIP-ITM** (for reranking based on generated captions).

## Prerequisites & Installation

Ensure you have the required Python packages installed. You can install the dependencies via pip:

```bash
pip install torch torchvision torchaudio numpy Pillow tqdm ultralytics open_clip_torch transformers hnswlib huggingface_hub
```

## Downloading the Models & Directory Layout

Because the fine-tuned models are large, they are hosted on Hugging Face. You must download them into a `models/` directory before running the evaluation script.

You can download them easily using the Hugging Face CLI:

```bash
# Create the models directory
mkdir models

# Download the model files from the Hugging Face repository
hf download sartma/vr_final_models clip_seed106.pt --local-dir models/
hf download sartma/vr_final_models best.pt --local-dir models/
hf download sartma/vr_final_models captions_cache.json --local-dir models/

# Rename the YOLO weights to match what the script expects
mv models/best.pt models/yolo_best.pt
```

After downloading, ensure your directory has the following structure:

```text
.
├── batch_eval.py
└── models/
    ├── clip_seed106.pt        <-- Fine-tuned CLIP checkpoint
    ├── yolo_best.pt           <-- Fine-tuned YOLO weights
    └── captions_cache.json    <-- Generated captions for the gallery images
```

## Usage

Run the script from the command line, providing the directories for the query and gallery images, along with the labels CSV.

```bash
python batch_eval.py \
  --query_dir /path/to/query_images \
  --gallery_dir /path/to/gallery_images \
  --labels /path/to/labels.csv \
  [--output eval_results.json] \
  [--cache_dir eval_cache]
```

### Arguments:
- `--query_dir` (Required): Path to the directory containing query images.
- `--gallery_dir` (Required): Path to the directory containing gallery images.
- `--labels` (Required): Path to a CSV file mapping image paths to item IDs.
- `--output` (Optional): The file path where the evaluation metric results will be saved in JSON format. (Default: `eval_results.json`)
- `--cache_dir` (Optional): Directory where gallery embeddings and metadata will be cached to speed up subsequent runs. (Default: `eval_cache`)

### Labels Format

The `--labels` CSV file should map the relative image path (or image name) to its unique item ID. **No header row is needed.**

**Example (`labels.csv`):**
```csv
img/MEN/Denim/id_00000001/01_1_front.jpg,id_00000001
img/MEN/Denim/id_00000001/01_2_side.jpg,id_00000001
img/MEN/Shirts/id_00000002/02_1_front.jpg,id_00000002
```

## How It Works

1. **Object Detection**: Passes the images through a fine-tuned YOLO model to crop the central clothing item, removing unnecessary background.
2. **Embedding**: Generates embeddings using a fine-tuned CLIP model (`ViT-B-32`). It fuses visual embeddings from CLIP and text embeddings generated from the provided `captions_cache.json`.
3. **Retrieval**: Uses `hnswlib` (Hierarchical Navigable Small World graphs) to perform fast Approximate Nearest Neighbor (ANN) search on the gallery embeddings.
4. **Reranking**: Uses `BLIP-ITM` (Image-Text Matching) to re-rank the top retrieved candidates (top 15) using higher-capacity cross-attention.
5. **Metrics Calculation**: Finally, it calculates Mean Average Precision (mAP), Normalized Discounted Cumulative Gain (NDCG), HitRate (HR), and Recall at K=5, 10, and 15.

## Caching Mechanism
The script automatically builds an HNSW index and caches the gallery embeddings in the `--cache_dir` (default: `eval_cache/`). This allows subsequent evaluations on the same gallery to start much faster. If you update the models or the gallery images, be sure to delete the cache directory to force it to rebuild.

## Kaggle Quickstart Example

To run a small test evaluation (e.g. 100 images) directly in a Kaggle Notebook using the DeepFashion dataset, you can paste the following snippet into a cell to automatically sample your dataset and generate the queries and gallery folders:

```python
import os, shutil, random
from pathlib import Path
from collections import defaultdict

dataset_root = Path('/kaggle/input/deepfashion-inshop-clothes-retrieval') 
eval_file = list(dataset_root.rglob("list_eval_partition.txt"))[0]

out_query = Path("/kaggle/working/query_images")
out_gallery = Path("/kaggle/working/gallery_images")
labels_csv = Path("/kaggle/working/labels.csv")
out_query.mkdir(parents=True, exist_ok=True)
out_gallery.mkdir(parents=True, exist_ok=True)

item_dict = defaultdict(lambda: {"query": [], "gallery": []})
with open(eval_file, "r") as f:
    for line in f.readlines()[2:]:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] in ["query", "gallery"]:
            item_dict[parts[1]][parts[2]].append(parts[0])

valid_items = [i for i, d in item_dict.items() if d["query"] and d["gallery"]]
random.seed(42)
random.shuffle(valid_items)

def find_image(img_path_str):
    matches = list(dataset_root.rglob(Path(img_path_str).name))
    return matches[0] if matches else None

labels_data = []
copied = 0
for item_id in valid_items:
    if copied >= 100: break
    for status in ["query", "gallery"]:
        out_dir = out_query if status == "query" else out_gallery
        for img in item_dict[item_id][status]:
            src = find_image(img)
            if src:
                dst = out_dir / Path(img).name
                shutil.copy(src, dst)
                labels_data.append(f"{dst.name},{item_id}\n")
                copied += 1

with open(labels_csv, "w") as f: f.writelines(labels_data)
print(f"Sampled {copied} images! Ready for evaluation.")
```

Then execute the script in the next cell:
```bash
!python batch_eval.py \
  --query_dir /kaggle/working/query_images \
  --gallery_dir /kaggle/working/gallery_images \
  --labels /kaggle/working/labels.csv
```
