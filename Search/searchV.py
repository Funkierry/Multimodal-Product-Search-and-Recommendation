import argparse
import os
import json
import sys
import faiss
import torch
import pandas as pd
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from tqdm import tqdm
import open_clip
from typing import List
from pathlib import Path

# Base Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fyp.artifacts import (
    resolve_image_path,
    validate_catalog_metadata,
    validate_normalized_vectors,
)

BASE_DIR = Path(os.environ.get('FYP_DATA_DIR', PROJECT_ROOT / 'dataset')).resolve()
SEARCH_DIR = Path(
    os.environ.get('PRODUCT_SEARCH_DIR', PROJECT_ROOT / 'Search')
).resolve()
os.makedirs(SEARCH_DIR, exist_ok=True)

# Data paths
TRAIN_CSV = os.path.join(BASE_DIR, 'train_updated.csv')
VAL_CSV = os.path.join(BASE_DIR, 'val_updated.csv')
TEST_CSV = os.path.join(BASE_DIR, 'test_updated.csv')
# Output paths
INDEX_PATH = os.path.join(SEARCH_DIR, "faiss_index.index")
META_JSON = os.path.join(SEARCH_DIR, "items_meta.json")
EMBEDDINGS_PATH = os.path.join(SEARCH_DIR, "image_embeddings.npy")

# Model config
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
BUILD_BATCH_SIZE = int(os.environ.get('FYP_EMBEDDING_BATCH_SIZE', '64'))


class DuplicateAsinError(ValueError):
    pass

def read_csv_safely(file_path: str) -> pd.DataFrame:
    print(f"[INFO] Reading CSV file: {os.path.basename(file_path)}")
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with tqdm(total=1, desc=f"Trying {encoding}") as pbar:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                pbar.update(1)
            print(f"[SUCCESS] Read with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {file_path} with any encoding")

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

def load_model():
    print("[INFO] Loading CLIP model...")
    try:
        with tqdm(total=1, desc="Loading model") as pbar:
            model, preprocess, _ = open_clip.create_model_and_transforms(
                MODEL_NAME,
                pretrained=PRETRAINED,
                device=DEVICE
            )
            tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            pbar.update(1)
        return model, preprocess, tokenizer
    except Exception as e:
        models, pretrains = open_clip.list_models()
        print("Available models:", models)
        print("Available pretrained weights:", pretrains)
        raise

def build_faiss_index(model, preprocess, tokenizer, csv_files: List[str], index_path=INDEX_PATH):
    print("[INFO] Building index from multiple datasets...")
    model.eval()

    # Combine all datasets
    all_dfs = []
    for csv_file in csv_files:
        df = read_csv_safely(csv_file)
        all_dfs.append(df)

    print("[INFO] Merging datasets...")
    with tqdm(total=1, desc="Merging data") as pbar:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        pbar.update(1)

    items_meta = []
    image_embeddings = []
    seen_asins = set()

    total_items = len(combined_df)
    print(f"[INFO] Processing {total_items} items...")

    with torch.inference_mode():
        progress_bar = tqdm(total=total_items, desc="Processing images", unit="img")
        for start in range(0, total_items, BUILD_BATCH_SIZE):
            batch = combined_df.iloc[start : start + BUILD_BATCH_SIZE]
            tensors = []
            valid_rows = []
            for _, row in batch.iterrows():
                try:
                    asin = row.get('asin')
                    if pd.isna(asin) or not str(asin).strip():
                        raise ValueError('ASIN is missing')
                    asin = str(asin).strip()
                    if asin in seen_asins:
                        raise DuplicateAsinError(f'Duplicate ASIN in source data: {asin}')
                    img_path, relative_img_path = resolve_image_path(row.get('imgUrl'), BASE_DIR)
                    with Image.open(img_path) as source:
                        tensors.append(preprocess(source.convert("RGB")))
                    valid_rows.append((row, asin, relative_img_path))
                    seen_asins.add(asin)
                except DuplicateAsinError:
                    raise
                except Exception as error:
                    progress_bar.write(f"Skipping {row.get('asin', 'unknown')}: {error}")
                finally:
                    progress_bar.update(1)

            if not tensors:
                continue
            image_batch = torch.stack(tensors).to(DEVICE)
            features = model.encode_image(image_batch)
            features = features / features.norm(dim=-1, keepdim=True)
            for (row, asin, relative_img_path), feature in zip(valid_rows, features.cpu().numpy()):
                image_embeddings.append(feature)
                meta = {}
                for key in [
                    'asin', 'title', 'imgUrl', 'productURL', 'price', 'stars',
                    'reviews', 'isBestSeller', 'boughtInLastMonth', 'categoryName'
                ]:
                    value = row.get(key, '')
                    meta[key] = '' if pd.isna(value) else value.item() if hasattr(value, 'item') else value
                meta['asin'] = asin
                meta['image_path'] = relative_img_path
                items_meta.append(meta)

        progress_bar.close()

    if not image_embeddings:
        raise RuntimeError("No valid images processed")

    print("[INFO] Converting embeddings to numpy array...")
    with tqdm(total=1, desc="Converting embeddings") as pbar:
        image_embeddings = np.array(image_embeddings, dtype=np.float32)
        pbar.update(1)
    validate_catalog_metadata(items_meta)
    validate_normalized_vectors(image_embeddings, 'search embeddings')

    print(f"[INFO] Building Faiss index with dimension {image_embeddings.shape[1]}...")
    with tqdm(total=2, desc="Building index") as pbar:
        index = faiss.IndexFlatIP(image_embeddings.shape[1])
        pbar.update(1)
        index.add(image_embeddings)
        pbar.update(1)

    print("[INFO] Saving files to:", SEARCH_DIR)
    with tqdm(total=2, desc="Saving files") as pbar:
        faiss.write_index(index, index_path)
        pbar.update(1)
        with open(META_JSON, "w", encoding="utf-8") as f:
            json.dump(items_meta, f, ensure_ascii=False, allow_nan=False)
        np.save(EMBEDDINGS_PATH, image_embeddings)
        pbar.update(1)

    print("[INFO] Index built successfully!")
    return index, items_meta

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_JSON):
        return None, None

    print("[INFO] Loading existing index...")
    with tqdm(total=2, desc="Loading files") as pbar:
        index = faiss.read_index(INDEX_PATH)
        pbar.update(1)
        with open(META_JSON, "r", encoding="utf-8") as f:
            items_meta = json.load(f)
        pbar.update(1)
    validate_catalog_metadata(items_meta)
    if index.ntotal != len(items_meta):
        raise ValueError('Search index and metadata counts differ; rebuild the search index')
    return index, items_meta

def search_products(query: str, model, tokenizer, index, items_meta, top_k=TOP_K):
    model.eval()
    with tqdm(total=3, desc="Searching") as pbar:
        with torch.no_grad():
            text = tokenizer([query]).to(DEVICE)
            text_features = model.encode_text(text)
            pbar.update(1)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype(np.float32)
            pbar.update(1)

        distances, indices = index.search(text_features, top_k)
        results = [
            (distances[0][i], items_meta[idx])
            for i, idx in enumerate(indices[0])
            if 0 <= idx < len(items_meta)
        ]
        pbar.update(1)

    return sorted(results, key=lambda x: x[0], reverse=True)

class ProductSearchGUI:
    def __init__(self, master, model, tokenizer, index, items_meta):
        self.master = master
        self.master.title("Product Search")

        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        search_frame = ttk.Frame(self.main_frame)
        search_frame.pack(fill="x", pady=5)

        ttk.Label(search_frame, text="Search Products:").pack(side="left")
        self.search_entry = ttk.Entry(search_frame, width=50)
        self.search_entry.pack(side="left", padx=5)
        ttk.Button(search_frame, text="Search", command=self.search).pack(side="left")

        self.results_container = ScrollableFrame(self.main_frame)
        self.results_container.pack(fill="both", expand=True, pady=5)

        self.model = model
        self.tokenizer = tokenizer
        self.index = index
        self.items_meta = items_meta

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame,
                                          variable=self.progress_var,
                                          maximum=100)
        self.progress_bar.pack(fill='x', pady=5)

    def display_result(self, result_frame, score, item):
        try:
            img_path, _ = resolve_image_path(item["image_path"], BASE_DIR)
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(img)
            img_label = ttk.Label(result_frame, image=photo)
            img_label.image = photo
            img_label.pack(side="left", padx=5)
        except Exception as e:
            print(f"Image error: {e}")

        info = (
            f"Score: {score:.3f}\n"
            f"Title: {item['title']}\n"
            f"Price: £{item['price']}\n"
            f"Category: {item['categoryName']}\n"
            f"Stars: {item['stars']} ({item['reviews']} reviews)\n"
            f"Best Seller: {'Yes' if item['isBestSeller'] else 'No'}"
        )
        ttk.Label(result_frame, text=info, justify="left").pack(side="left", padx=5)

    def search(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query")
            return

        self.progress_var.set(0)

        for widget in self.results_container.scrollable_frame.winfo_children():
            widget.destroy()

        self.progress_var.set(30)
        self.master.update()

        results = search_products(query, self.model, self.tokenizer, self.index, self.items_meta)

        self.progress_var.set(70)
        self.master.update()

        if not results:
            ttk.Label(self.results_container.scrollable_frame,
                     text="No results found").pack(pady=10)
            self.progress_var.set(100)
            return

        for score, item in results:
            result_frame = ttk.Frame(self.results_container.scrollable_frame)
            result_frame.pack(fill="x", pady=5, padx=5)
            self.display_result(result_frame, score, item)

        self.progress_var.set(100)

def main():
    parser = argparse.ArgumentParser(description="Build or run the product search index")
    parser.add_argument('--rebuild', action='store_true', help='Rebuild search artifacts')
    parser.add_argument('--build-only', action='store_true', help='Exit after building/loading')
    args = parser.parse_args()
    print("[INFO] Starting product search system...")

    # Create search directory if not exists
    os.makedirs(SEARCH_DIR, exist_ok=True)
    print(f"[INFO] Search directory: {SEARCH_DIR}")

    # Load model
    model, preprocess, tokenizer = load_model()

    # Process all datasets
    csv_files = [TRAIN_CSV, VAL_CSV, TEST_CSV]

    # Load or build index
    if args.rebuild:
        index, items_meta = build_faiss_index(model, preprocess, tokenizer, csv_files)
    else:
        index, items_meta = load_index()
    if index is None:
        print("[INFO] No existing index found, building new index...")
        index, items_meta = build_faiss_index(model, preprocess, tokenizer, csv_files)

    if args.build_only:
        return

    # Start GUI
    root = ThemedTk(theme="arc")
    root.geometry("800x600")
    app = ProductSearchGUI(root, model, tokenizer, index, items_meta)
    root.mainloop()

if __name__ == "__main__":
    main()
