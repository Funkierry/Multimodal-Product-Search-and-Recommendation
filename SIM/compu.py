import os

# --------------------------------------------------------------------
# 1) 解决 OMP 多次初始化的问题
# --------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import traceback
import faiss
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import open_clip
import warnings
from pathlib import Path
import hashlib

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

class EmbeddingBuilder:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        self.META_JSON = Path(
            os.environ.get('FYP_ITEMS_META', project_root / 'Search' / 'items_meta.json')
        ).resolve()
        self.IMAGES_DIR = Path(
            os.environ.get('FYP_IMAGE_DIR', project_root / 'dataset' / 'images')
        ).resolve()
        self.SAVE_DIR = Path(
            os.environ.get('PRODUCT_CONTENT_DIR', project_root / 'SIM')
        ).resolve()

        # 输出的 FAISS 索引文件 和 embeddings 矩阵文件
        self.INDEX_PATH = os.path.join(self.SAVE_DIR, "faiss_index_sim.index")
        self.EMB_PATH   = os.path.join(self.SAVE_DIR, "embeddings_sim.npy")

        # 设备
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化空变量
        self.model = None
        self.tokenizer = None
        self.items_meta = []
        self.embeddings = []

        self.init_clip_model()
        self.load_data()

    def init_clip_model(self):
        """加载 OpenCLIP 模型和 tokenizer。"""
        try:
            self.model, _, self.img_transform = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained='openai',
                device=self.DEVICE
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
            self.model.eval()
            print("Successfully initialized OpenCLIP model.")
        except Exception as e:
            print(f"Error loading OpenCLIP model: {e}")
            traceback.print_exc()
            raise

    def load_data(self):
        """加载 items_meta.json 并规范化图像路径。"""
        if not os.path.exists(self.META_JSON):
            raise FileNotFoundError(f"{self.META_JSON} not found.")

        with open(self.META_JSON, 'r', encoding='utf-8') as f:
            self.items_meta = json.load(f)
        print(f"Loaded items_meta with {len(self.items_meta)} items.")

        # 处理 image_path，使其位于 self.IMAGES_DIR 下
        for idx, item in enumerate(self.items_meta):
            original_path = item.get('image_path', "")
            if not original_path:
                continue
            if os.path.isabs(original_path):
                # 如果是绝对路径，但是不在 IMAGES_DIR 下，仅做提示
                if not os.path.normpath(original_path).lower().startswith(
                    os.path.normpath(self.IMAGES_DIR).lower()
                ):
                    print(f"[Warning] item {idx}: {original_path} not under IMAGES_DIR.")
                item['image_path'] = os.path.normpath(original_path)
            else:
                # 相对路径 => 拼接到 IMAGES_DIR
                new_path = os.path.join(self.IMAGES_DIR, original_path)
                item['image_path'] = os.path.normpath(new_path)

    def compute_embeddings(self):
        """计算图像 + 文本融合向量，并存到 self.embeddings."""
        self.embeddings = [None] * len(self.items_meta)
        batch_size = int(os.environ.get('FYP_EMBEDDING_BATCH_SIZE', '64'))
        batch_indices = []
        image_tensors = []
        titles = []

        def flush_batch():
            if not batch_indices:
                return
            image_batch = torch.stack(image_tensors).to(self.DEVICE)
            text_tokens = self.tokenizer(titles).to(self.DEVICE)
            with torch.inference_mode():
                image_features = self.model.encode_image(image_batch)
                text_features = self.model.encode_text(text_tokens)
                fused = 0.5 * image_features + 0.5 * text_features
                fused = fused / fused.norm(dim=-1, keepdim=True)
            for item_index, embedding in zip(batch_indices, fused.cpu().numpy()):
                self.embeddings[item_index] = embedding
            batch_indices.clear()
            image_tensors.clear()
            titles.clear()

        for idx, item in enumerate(tqdm(self.items_meta, desc="Computing embeddings")):
            img_path = item.get('image_path', "")
            if not img_path or not os.path.exists(img_path):
                continue

            title = item.get('title', "")

            try:
                with Image.open(img_path) as source:
                    image_tensors.append(self.img_transform(source.convert('RGB')))
                batch_indices.append(idx)
                titles.append(title)
                if len(batch_indices) >= batch_size:
                    flush_batch()
            except Exception as e:
                print(f"[Error] item {idx}, path={img_path}, title={title}: {e}")
                traceback.print_exc()
        flush_batch()

    def build_faiss_index(self):
        """将 self.embeddings 写入 FAISS 索引并保存到磁盘。"""
        valid_embeddings = [x for x in self.embeddings if x is not None]
        if not valid_embeddings:
            print("No valid embeddings to build index.")
            return

        d = valid_embeddings[0].shape[0]   # 向量维度
        index = faiss.IndexFlatIP(d)       # InnerProduct => 归一化后可视为余弦相似

        full_embeddings = [
            embedding if embedding is not None else np.zeros(d, dtype=np.float32)
            for embedding in self.embeddings
        ]
        emb_matrix = np.asarray(full_embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_matrix)
        index.add(emb_matrix)

        # 保存索引
        faiss.write_index(index, self.INDEX_PATH)
        print(f"FAISS index built with {index.ntotal} vectors => {self.INDEX_PATH}")

        np.save(self.EMB_PATH, emb_matrix)
        print(f"Saved embeddings to {self.EMB_PATH}")

        digest = hashlib.sha256()
        with self.META_JSON.open('rb') as metadata_source:
            while chunk := metadata_source.read(1024 * 1024):
                digest.update(chunk)
        dataset_hash = digest.hexdigest()
        search_index = self.META_JSON.parent / 'faiss_index.index'
        manifest = {
            'schema_version': 1,
            'dataset_sha256': dataset_hash,
            'model_name': 'ViT-L-14',
            'pretrained': 'openai',
            'embedding_dimension': d,
            'item_count': len(self.items_meta),
            'metadata_path': os.path.relpath(self.META_JSON, self.SAVE_DIR),
            'search_index_path': os.path.relpath(search_index, self.SAVE_DIR),
            'content_index_path': os.path.basename(self.INDEX_PATH),
            'content_embeddings_path': os.path.basename(self.EMB_PATH),
        }
        manifest_path = self.SAVE_DIR / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f"Saved artifact manifest to {manifest_path}")

    def run(self):
        """一次性执行：Compute Embeddings + Build Index."""
        self.compute_embeddings()
        self.build_faiss_index()


if __name__ == "__main__":
    builder = EmbeddingBuilder()
    os.makedirs(builder.SAVE_DIR, exist_ok=True)
    builder.run()
    print("Done. Embeddings + FAISS index saved.")
