"""
batch_eval.py — Batch Evaluation for Visual Product Search
==========================================================
Computes Recall@K, NDCG@K, mAP@K (K ∈ {5, 10, 15}) using the best model.

Expected layout
---------------
models/
  clip_seed106.pt        ← fine-tuned CLIP checkpoint (best seed)
  yolo_best.pt           ← fine-tuned YOLO weights
  captions_cache.json   ← BLIP-2 captions for gallery items

Usage
-----
python batch_eval.py \
  --query_dir   /path/to/query_images \
  --gallery_dir /path/to/gallery_images \
  --labels      labels.csv \
  [--output     results.json]

labels.csv format (no header needed):
  image_name,item_id
  img/MEN/Denim/id_00000001/01_1_front.jpg,id_00000001
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ── Hardcoded config ────────────────────────────────────────────────────────
CLIP_CKPT      = "models/clip_seed106.pt"
YOLO_CKPT      = "models/yolo_best.pt"
CAPTIONS_FILE  = "models/captions_cache.json"
ALPHA          = 0.7          # gallery fusion weight  (image: 0.7, text: 0.3)
K_VALUES       = [5, 10, 15]
RERANK_K       = 15
YOLO_PAD       = 10
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query_dir",   required=True)
    p.add_argument("--gallery_dir", required=True)
    p.add_argument("--labels",      required=True,  help="CSV: image_name,item_id")
    p.add_argument("--output",      default="eval_results.json")
    p.add_argument("--cache_dir",   default="eval_cache")
    return p.parse_args()


# ── Models ──────────────────────────────────────────────────────────────────
def load_models(device):
    from ultralytics import YOLO
    import open_clip
    from transformers import BlipProcessor, BlipForImageTextRetrieval

    print(f"[YOLO]  {YOLO_CKPT}")
    yolo = YOLO(YOLO_CKPT)

    print(f"[CLIP]  ViT-B-32 <- {CLIP_CKPT}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model.load_state_dict(torch.load(CLIP_CKPT, map_location="cpu"))
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer("ViT-B-32")

    print("[BLIP]  blip-itm-base-coco")
    itm_proc  = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    itm_model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco", torch_dtype=torch.float16
    ).to(device).eval()

    return yolo, clip_model, preprocess, tokenizer, itm_proc, itm_model


# ── Embeddings ───────────────────────────────────────────────────────────────
def yolo_crop(path, yolo, pad=YOLO_PAD):
    img = Image.open(path).convert("RGB")
    try:
        boxes = yolo(path, verbose=False)[0].boxes
        if len(boxes) == 0:
            return img
        best = max(boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        W, H = img.size
        return img.crop((max(0,x1-pad), max(0,y1-pad), min(W,x2+pad), min(H,y2+pad)))
    except Exception:
        return img

def img_emb(pil, model, preprocess, device):
    x = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model.encode_image(x)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().float().numpy()[0]

def txt_emb(text, model, tokenizer, device):
    t = tokenizer([text]).to(device)
    with torch.no_grad():
        e = model.encode_text(t)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().float().numpy()[0]

def fuse(ie, te):
    v = ALPHA * ie + (1 - ALPHA) * te
    return v / (np.linalg.norm(v) + 1e-8)


# ── Labels & image collection ────────────────────────────────────────────────
def load_labels(csv_path):
    mapping = {}
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) != 2 or parts[0].lower() == "image_name":
                continue
            mapping[parts[0].strip()] = parts[1].strip()
    print(f"[Labels] {len(mapping)} entries")
    return mapping

def collect_images(folder, label_map):
    entries = []
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, folder).replace("\\", "/")
            item_id  = label_map.get(rel_path) or label_map.get(fname)
            if item_id is None:
                for k, v in label_map.items():
                    if k.endswith("/" + fname):
                        item_id = v; break
            if item_id:
                entries.append((abs_path, rel_path, item_id))
    return entries


# ── Gallery index ────────────────────────────────────────────────────────────
def build_gallery_index(entries, yolo, clip_model, preprocess, tokenizer,
                        device, captions, cache_dir):
    import hnswlib
    os.makedirs(cache_dir, exist_ok=True)
    emb_path   = os.path.join(cache_dir, "gallery_embs.npy")
    meta_path  = os.path.join(cache_dir, "gallery_meta.json")
    index_path = os.path.join(cache_dir, "gallery.hnsw")

    if os.path.exists(emb_path) and os.path.exists(meta_path):
        embs = np.load(emb_path)
        with open(meta_path) as f: meta = json.load(f)
        print(f"[Gallery] Loaded {embs.shape[0]} embeddings from cache")
    else:
        embs, meta = [], []
        for abs_path, rel_path, item_id in tqdm(entries, desc="[Gallery] Embedding"):
            try:
                crop = yolo_crop(abs_path, yolo)
                ie   = img_emb(crop, clip_model, preprocess, device)
                if captions:
                    cap   = captions.get(rel_path, "clothing item")
                    final = fuse(ie, txt_emb(cap, clip_model, tokenizer, device))
                else:
                    final = ie
                embs.append(final)
                meta.append({"idx": len(embs)-1, "rel_path": rel_path,
                             "abs_path": abs_path, "item_id": item_id})
            except Exception as ex:
                print(f"  skip {rel_path}: {ex}")
        embs = np.array(embs, dtype=np.float32)
        np.save(emb_path, embs)
        with open(meta_path, "w") as f: json.dump(meta, f)
        print(f"[Gallery] {embs.shape[0]} embeddings saved")

    idx = hnswlib.Index(space="cosine", dim=embs.shape[1])
    if os.path.exists(index_path):
        idx.load_index(index_path)
    else:
        idx.init_index(max_elements=len(embs), ef_construction=200, M=16)
        idx.add_items(embs, np.arange(len(embs)))
        idx.save_index(index_path)
    idx.set_ef(50)
    return idx, meta


# ── ITM rerank ───────────────────────────────────────────────────────────────
def itm_rerank(query_crop, cands, meta, captions, itm_proc, itm_model, device):
    if captions is None:
        return cands
    valid_caps, valid_pos = [], []
    for pos, c in enumerate(cands):
        cap = captions.get(meta[c]["rel_path"], "")
        if cap:
            valid_caps.append(cap)
            valid_pos.append(pos)
    if not valid_caps:
        return cands
    try:
        inp = itm_proc(images=[query_crop]*len(valid_caps), text=valid_caps,
                       return_tensors="pt", padding=True).to(device)
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            scores = F.softmax(itm_model(**inp, use_itm_head=True).itm_score, dim=1)[:,1].cpu().numpy()
    except Exception:
        return cands
    final = np.zeros(len(cands))
    for s, pos in zip(scores, valid_pos):
        final[pos] = s
    return [cands[i] for i in np.argsort(-final, kind="stable")]


# ── Metrics ──────────────────────────────────────────────────────────────────
def hit_at_k(ret, rel_id, k, nr=None):
    return 1.0 if rel_id in ret[:k] else 0.0

def recall_at_k(ret, rel_id, k, nr):
    return sum(1 for r in ret[:k] if r == rel_id) / nr if nr else 0.0

def ndcg_at_k(ret, rel_id, k, nr):
    dcg   = sum(1/np.log2(i+2) for i,r in enumerate(ret[:k]) if r == rel_id)
    ideal = sum(1/np.log2(i+2) for i in range(min(nr, k)))
    return dcg/ideal if ideal else 0.0

def map_at_k(ret, rel_id, k, nr):
    h, s = 0, 0.0
    for i, r in enumerate(ret[:k]):
        if r == rel_id: h += 1; s += h/(i+1)
    return s/nr if nr else 0.0


# ── Eval loop ────────────────────────────────────────────────────────────────
def run_eval(query_entries, idx, meta, gallery_item_counts,
             yolo, clip_model, preprocess, device,
             itm_proc, itm_model, captions):
    scores = {k: {"hitrate": [], "recall": [], "ndcg": [], "map": []} for k in K_VALUES}
    fetch_k = max(RERANK_K, max(K_VALUES))

    for abs_path, rel_path, item_id in tqdm(query_entries, desc="[Eval]"):
        try:
            crop = yolo_crop(abs_path, yolo)
            qe   = img_emb(crop, clip_model, preprocess, device)   # vision-only query
            labels, _ = idx.knn_query(qe.reshape(1,-1), k=fetch_k)
            rl   = itm_rerank(crop, list(labels[0]), meta, captions, itm_proc, itm_model, device)
            ret  = [meta[l]["item_id"] for l in rl]
            nr   = gallery_item_counts.get(item_id, 1)
            for k in K_VALUES:
                scores[k]["hitrate"].append(hit_at_k(  ret, item_id, k))
                scores[k]["recall"].append(recall_at_k(ret, item_id, k, nr))
                scores[k]["ndcg"].append(  ndcg_at_k(  ret, item_id, k, nr))
                scores[k]["map"].append(   map_at_k(   ret, item_id, k, nr))
        except Exception as ex:
            print(f"  skip query {rel_path}: {ex}")
    return scores


# ── Report ───────────────────────────────────────────────────────────────────
def print_report(scores):
    w = 12 + 9*len(K_VALUES)
    print("\n" + "="*w)
    print("RESULTS")
    print("="*w)
    print(f"{'':12}" + "".join(f"  K={k:<5}" for k in K_VALUES))
    print("-"*w)
    for m, label in [("hitrate","HitRate@K"), ("recall","Recall@K"), ("ndcg","NDCG@K"), ("map","mAP@K")]:
        row = f"{label:<12}" + "".join(f"  {np.mean(scores[k][m]):.4f}" for k in K_VALUES)
        print(row)
    print("="*w)
    for m, label in [("hitrate","HitRate@K"), ("recall","Recall@K"), ("ndcg","NDCG@K"), ("map","mAP@K")]:
        print(f"\n  {label}")
        for k in K_VALUES:
            v = scores[k][m]
            print(f"    K={k:2d}: {np.mean(v):.4f} +/- {np.std(v):.4f}  (n={len(v)})")


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}\n")

    for ckpt in [CLIP_CKPT, YOLO_CKPT, CAPTIONS_FILE]:
        if not os.path.exists(ckpt):
            sys.exit(f"Missing: {ckpt}\nMake sure all files are in the models/ folder.")

    label_map = load_labels(args.labels)

    query_entries   = collect_images(args.query_dir,   label_map)
    gallery_entries = collect_images(args.gallery_dir, label_map)
    print(f"[Images] {len(query_entries)} queries, {len(gallery_entries)} gallery")

    if not query_entries or not gallery_entries:
        sys.exit("No labeled images found. Check paths and labels.csv.")

    from collections import Counter
    gallery_item_counts = Counter(item_id for _, _, item_id in gallery_entries)

    if not os.path.exists(CAPTIONS_FILE):
        sys.exit(f"Missing: {CAPTIONS_FILE}\nPlace captions_cache.json in the models/ folder.")
    with open(CAPTIONS_FILE) as f: captions = json.load(f)
    print(f"[Captions] {len(captions)} entries loaded")

    yolo, clip_model, preprocess, tokenizer, itm_proc, itm_model = load_models(device)

    idx, meta = build_gallery_index(
        gallery_entries, yolo, clip_model, preprocess, tokenizer,
        device, captions, args.cache_dir
    )

    scores = run_eval(
        query_entries, idx, meta, gallery_item_counts,
        yolo, clip_model, preprocess, device,
        itm_proc, itm_model, captions
    )

    print_report(scores)

    out = {str(k): {m: float(np.mean(scores[k][m])) for m in ["recall","ndcg","map"]}
           for k in K_VALUES}
    with open(args.output, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[Saved] {args.output}")


if __name__ == "__main__":
    main()