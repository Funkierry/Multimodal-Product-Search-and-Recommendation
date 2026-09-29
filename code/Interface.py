import os
import sys
import json
import faiss
import numpy as np
import torch
import tkinter as tk
import webbrowser
from PIL import Image, ImageTk
import open_clip
import customtkinter as ctk
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import spacy
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fyp.artifacts import ArtifactManifest, resolve_image_path
from fyp.config import ProjectPaths, search_candidate_count
from fyp.recommendation.collaborative import InteractionRecommender
from fyp.scoring import similarity_percent

PROJECT_PATHS = ProjectPaths.from_environment()


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def interpolate_color(start_color, end_color, fraction):
    try:
        start_color = start_color.lstrip('#')
        end_color = end_color.lstrip('#')
        sr, sg, sb = int(start_color[0:2], 16), int(start_color[2:4], 16), int(start_color[4:6], 16)
        er, eg, eb = int(end_color[0:2], 16), int(end_color[2:4], 16), int(end_color[4:6], 16)

        r = int(sr + (er - sr) * fraction)
        g = int(sg + (eg - sg) * fraction)
        b = int(sb + (eb - sb) * fraction)

        # Clamp values to 0-255 range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        # Fallback to end_color if interpolation fails
        return f"#{end_color.lstrip('#')}"


# Disclaimer window class
class DisclaimerWindow(ctk.CTkToplevel):
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Disclaimer")
        self.geometry("600x350")

        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = 600
        window_height = 350
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.attributes("-alpha", 0.0)
        self.fade_in_step = 0.0
        self.build_ui()
        self.grab_set()
        self.focus()
        self.after(10, self.fade_in)

    def build_ui(self):
        label = ctk.CTkLabel(
            self,
            text=(
                "【 Disclaimer】\n\n"
                "This system is for demonstration/educational purposes only.\n"
                "We do not accept any liability for consequences.\n\n"
                "By clicking 'Confirm', you acknowledge and agree."
            ),
            font=("Arial Rounded MT Bold", 14),
            wraplength=550,
            justify="left"
        )
        label.pack(padx=20, pady=20, fill="both", expand=True)

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=(0, 20))

        confirm_button = ctk.CTkButton(
            button_frame,
            text="Confirm / 确认",
            command=self.confirm_action,
            corner_radius=10,
            font=("Arial Rounded MT Bold", 14),
            fg_color="#FFA41C",
            hover_color="#E08E0B",
            width=120
        )
        confirm_button.pack()


    def fade_in(self):
        if self.fade_in_step < 1.0:
            self.fade_in_step += 0.05
            self.attributes("-alpha", self.fade_in_step)
            self.after(50, self.fade_in)
        else:
             self.attributes("-alpha", 1.0)

    def confirm_action(self):
        self.destroy()


# RangeSlider class (Price range slider)
class RangeSlider(ctk.CTkFrame):

    def __init__(self, parent, total_range=(0, 1000), on_change=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(height=28, corner_radius=14)
        self.pack_propagate(False)

        self.min_val, self.max_val = total_range
        self.current_left = self.min_val
        self.current_right = self.max_val
        self.on_change = on_change
        self.knob_radius = 9
        self.canvas_bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        self.track_color = "#FF9900"
        self.knob_color = "#FF9900"
        self.canvas_bg_color_actual = self._apply_appearance_mode(("#DBDBDB", "#252525"))

        self.canvas = tk.Canvas(
            self,
            bg=self.canvas_bg_color_actual,
            highlightthickness=0,
            height=28
        )
        self.canvas.pack(fill="x", expand=True)

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.bind_events()
        self.after(10, self.redraw)

    def on_canvas_resize(self, event):
        self.redraw()

    def bind_events(self):
        self.canvas.bind("<Button-1>", self.click_event)
        self.canvas.bind("<B1-Motion>", self.drag_event)
        self.canvas.bind("<ButtonRelease-1>", self.release_event)

    def redraw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1: return
        cy = h // 2
        line_pad = self.knob_radius + 2
        line_x1 = line_pad
        line_x2 = w - line_pad
        if line_x2 <= line_x1: return

        total_range_span = self.max_val - self.min_val
        if total_range_span <= 0: total_range_span = 1

        self.current_left = max(self.min_val, min(self.current_left, self.max_val))
        self.current_right = max(self.min_val, min(self.current_right, self.max_val))

        if total_range_span == 0:
            left_ratio = 0.0
            right_ratio = 1.0
        else:
            left_ratio = (self.current_left - self.min_val) / total_range_span
            right_ratio = (self.current_right - self.min_val) / total_range_span

        left_px = line_x1 + (line_x2 - line_x1) * left_ratio
        right_px = line_x1 + (line_x2 - line_x1) * right_ratio

        track_bg_color = self._apply_appearance_mode(("#B0B0B0", "#404040"))
        self.canvas.create_line(line_x1, cy, line_x2, cy, width=4, fill=track_bg_color, smooth=True)
        self.canvas.create_line(left_px, cy, right_px, cy, width=5, fill=self.track_color, smooth=True)
        self.canvas.create_oval(
            left_px - self.knob_radius, cy - self.knob_radius,
            left_px + self.knob_radius, cy + self.knob_radius,
            fill=self.knob_color, outline=self.knob_color
        )
        self.canvas.create_oval(
            right_px - self.knob_radius, cy - self.knob_radius,
            right_px + self.knob_radius, cy + self.knob_radius,
            fill=self.knob_color, outline=self.knob_color
        )

    def click_event(self, event):
        self._update_dragging_knob(event.x)
        if self.dragging_knob:
            self.move_knob(event.x)

    def drag_event(self, event):
        if hasattr(self, 'dragging_knob') and self.dragging_knob:
            self.move_knob(event.x)

    def release_event(self, event):
         self.dragging_knob = None

    def _update_dragging_knob(self, x):
        self.dragging_knob = None
        w = self.canvas.winfo_width()
        line_pad = self.knob_radius + 2
        line_x1 = line_pad
        line_x2 = w - line_pad
        if line_x2 <= line_x1: return

        total_range_span = self.max_val - self.min_val
        if total_range_span <= 0: total_range_span = 1

        self.current_left = max(self.min_val, min(self.current_left, self.max_val))
        self.current_right = max(self.min_val, min(self.current_right, self.max_val))

        if total_range_span == 0:
            left_ratio = 0.0
            right_ratio = 1.0
        else:
            left_ratio = (self.current_left - self.min_val) / total_range_span
            right_ratio = (self.current_right - self.min_val) / total_range_span

        left_px = line_x1 + (line_x2 - line_x1) * left_ratio
        right_px = line_x1 + (line_x2 - line_x1) * right_ratio
        dist_left = abs(x - left_px)
        dist_right = abs(x - right_px)
        click_tolerance = self.knob_radius * 1.5

        if dist_left <= click_tolerance and dist_left < dist_right:
            self.dragging_knob = "left"
        elif dist_right <= click_tolerance:
            self.dragging_knob = "right"
        elif left_px < x < right_px:
             self.dragging_knob = "left" if dist_left < dist_right else "right"
        elif x < left_px and dist_left <= click_tolerance:
             self.dragging_knob = "left"
        elif x > right_px and dist_right <= click_tolerance:
             self.dragging_knob = "right"

    def move_knob(self, x):
        if not hasattr(self, 'dragging_knob') or self.dragging_knob is None: return

        w = self.canvas.winfo_width()
        line_pad = self.knob_radius + 2
        line_x1 = line_pad
        line_x2 = w - line_pad
        if line_x2 <= line_x1: return

        total_range_span = self.max_val - self.min_val
        if total_range_span <= 0 or line_x2 <= line_x1 :
            val = self.current_left if self.dragging_knob == "left" else self.current_right
        else:
            x_clamped = max(line_x1, min(x, line_x2))
            ratio = (x_clamped - line_x1) / (line_x2 - line_x1)
            val = self.min_val + ratio * total_range_span

        target = self.dragging_knob
        new_left = self.current_left
        new_right = self.current_right

        if target == "left":
            new_left = min(val, self.current_right)
            new_left = max(self.min_val, new_left)
        elif target == "right":
            new_right = max(val, self.current_left)
            new_right = min(self.max_val, new_right)

        tolerance = 1e-6
        if abs(new_left - self.current_left) > tolerance or abs(new_right - self.current_right) > tolerance:
            self.current_left = round(new_left, 2)
            self.current_right = round(new_right, 2)
            self.redraw()
            if self.on_change:
                self.on_change(self.current_left, self.current_right)


# CSV processing: adjective extraction + emotion scoring
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    print(
        "spaCy model 'en_core_web_sm' is unavailable. "
        "Run 'python -m spacy download en_core_web_sm' to rebuild review summaries."
    )

analyzer = SentimentIntensityAnalyzer()

def sentiment_analysis_vader(review: str) -> int:
    if not isinstance(review, str): return 0
    sentiment_score = analyzer.polarity_scores(review)['compound']
    if sentiment_score <= -0.05: return -1
    elif sentiment_score >= 0.05: return 1
    else: return 0

def extract_adjectives_spacy(review: str):
    if not isinstance(review, str) or nlp is None: return []
    doc = nlp(review.lower())
    adjectives = [token.lemma_ for token in doc if token.pos_ == 'ADJ' and not token.is_stop]
    return adjectives

def build_reviews_data(csv_path: str, output_json_path: str):

    print(f"Attempting to build reviews data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file doesn't exist: {csv_path}")
        return

    try:
        data = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f"Successfully loaded CSV. Columns: {data.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    required_cols = ["ASIN", "user", "Review"]
    if not all(col in data.columns for col in required_cols):
        print(f"Error: CSV necessary columns are missing in it. Need: {required_cols}, In fact, there is: {data.columns.tolist()}")
        found_cols = {col.lower(): col for col in data.columns}
        corrected_map = {}
        missing = []
        for req_col in required_cols:
             if req_col.lower() in found_cols:
                  corrected_map[req_col] = found_cols[req_col.lower()]
             else:
                  missing.append(req_col)
        if len(missing) == 0:
            print(f"Found columns with different casing. Renaming: {corrected_map}")
            data.rename(columns={v: k for k, v in corrected_map.items()}, inplace=True)
        else:
             print(f"Still missing: {missing}. Please check the CSV structure.")
             return

    print("Cleaning data...")
    initial_rows = len(data)
    data.dropna(subset=['ASIN', 'Review'], inplace=True)
    data['Review'] = data['Review'].astype(str)
    data.drop_duplicates(subset=['ASIN', 'user', 'Review'], inplace=True)
    print(f"Removed {initial_rows - len(data)} rows with missing ASIN/Review or duplicates.")

    print("Performing sentiment analysis...")
    tqdm.pandas(desc="Analyzing Sentiment")
    data['Sentiment'] = data['Review'].progress_apply(sentiment_analysis_vader)

    print("Grouping reviews by ASIN...")
    asin_groups = data.groupby('ASIN')
    reviews_data = {}
    num_asins = len(asin_groups)

    print(f"Extracting adjectives and summarizing for {num_asins} ASINs...")
    for asin, group in tqdm(asin_groups, total=num_asins, desc="Processing ASINs"):
        all_adjectives = []
        positive_reviews_adjs = []
        negative_reviews_adjs = []

        for _, row in group.iterrows():
            adjs = extract_adjectives_spacy(row['Review'])
            if row['Sentiment'] == 1:
                positive_reviews_adjs.extend(adjs)
            elif row['Sentiment'] == -1:
                negative_reviews_adjs.extend(adjs)
            all_adjectives.extend(adjs)

        pos_adj_counter = Counter(positive_reviews_adjs)
        neg_adj_counter = Counter(negative_reviews_adjs)
        top_n = 6
        top_pos = pos_adj_counter.most_common(top_n)
        top_neg = neg_adj_counter.most_common(top_n)
        aspects = list(dict.fromkeys([adj for adj, freq in top_pos] + [adj for adj, freq in top_neg]))
        if not aspects: continue

        positive_scores = []
        negative_scores = []
        for aspect in aspects:
            positive_scores.append(pos_adj_counter.get(aspect, 0))
            negative_scores.append(neg_adj_counter.get(aspect, 0))

        positive_keywords = [adj for adj, freq in top_pos for _ in range(freq)]
        negative_keywords = [adj for adj, freq in top_neg for _ in range(freq)]

        reviews_data[asin] = {
            "aspects": aspects,
            "positiveScores": positive_scores,
            "negativeScores": negative_scores,
            "positiveKeywords": positive_keywords,
            "negativeKeywords": negative_keywords
        }

    print(f"Saving processed data to {output_json_path}")
    try:
        with open(output_json_path, "w", encoding="utf-8") as jf:
            json.dump(reviews_data, jf, ensure_ascii=False, indent=2)
        print(f"Successfully generated reviews_data.json => {output_json_path}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")


# The search section of Faiss + CLIP
class ProductSearchSystem:
    def __init__(self):
        self.SEARCH_DIR = Path(
            os.environ.get('PRODUCT_SEARCH_DIR', PROJECT_ROOT / 'Search')
        ).expanduser().resolve()
        self.INDEX_PATH = os.path.join(self.SEARCH_DIR, "faiss_index.index")
        self.META_JSON = os.path.join(self.SEARCH_DIR, "items_meta.json")
        self.CONTENT_DIR = Path(
            os.environ.get('PRODUCT_CONTENT_DIR', PROJECT_ROOT / 'SIM')
        ).expanduser().resolve()
        self.CONTENT_INDEX_PATH = self.CONTENT_DIR / "faiss_index_sim.index"
        self.EMB_PATH = self.CONTENT_DIR / "embeddings_sim.npy"
        self.MANIFEST_PATH = self.CONTENT_DIR / "manifest.json"

        self.MODEL_NAME = "ViT-L-14"
        self.PRETRAINED = "openai"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.DEVICE}")
        self.TOP_K = search_candidate_count()

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.index = None
        self.content_index = None
        self.items_meta = []
        self.embeddings = None
        self.load_success = False

        try:
            self.load_data()
            self.init_model()
            self.load_success = True
            # Create ASIN to index mapping for faster lookups
            self.asin_to_metadata_map = {item.get('asin') or item.get('ASIN'): item
                                         for item in self.items_meta
                                         if item.get('asin') or item.get('ASIN')}
            self.asin_to_index_map = {
                item.get('asin') or item.get('ASIN'): index
                for index, item in enumerate(self.items_meta)
                if item.get('asin') or item.get('ASIN')
            }
            def popularity_key(item):
                try:
                    rating = float(item.get('stars', 0))
                except (TypeError, ValueError):
                    rating = 0.0
                try:
                    reviews = int(float(item.get('reviews', 0)))
                except (TypeError, ValueError):
                    reviews = 0
                return rating, reviews

            self.popular_items = sorted(
                self.items_meta, key=popularity_key, reverse=True
            )
            self.items_by_category = {}
            for metadata in self.popular_items:
                category = str(metadata.get('categoryName', 'Unknown')).casefold()
                self.items_by_category.setdefault(category, []).append(metadata)
            print(f"Created ASIN to metadata map with {len(self.asin_to_metadata_map)} entries.")
        except Exception as e:
            print("="*30)
            print("!!! FAILED TO INITIALIZE ProductSearchSystem !!!")
            print(f"Error: {e}")
            print("Please check model availability, file paths, and dependencies.")
            print("Search functionality will be disabled.")
            print("="*30)
            traceback.print_exc()

    def init_model(self):
        print(f"Initializing CLIP model: {self.MODEL_NAME} ({self.PRETRAINED})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME,
            pretrained=self.PRETRAINED,
            device=self.DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self.model.eval()
        print("Successfully initialized model and tokenizer.")

    def load_data(self):
        print("Loading search index, metadata, and embeddings...")
        if not self.MANIFEST_PATH.is_file():
            raise FileNotFoundError(
                f"Artifact manifest not found: {self.MANIFEST_PATH}. "
                "Rebuild with Search/searchV.py and SIM/compu.py."
            )
        manifest = ArtifactManifest.load(self.MANIFEST_PATH)
        self.MODEL_NAME = manifest.model_name
        self.PRETRAINED = manifest.pretrained
        self.INDEX_PATH = str(self.CONTENT_DIR / manifest.search_index_path)
        self.META_JSON = str(self.CONTENT_DIR / manifest.metadata_path)
        self.CONTENT_INDEX_PATH = self.CONTENT_DIR / manifest.content_index_path
        self.EMB_PATH = self.CONTENT_DIR / manifest.content_embeddings_path
        if not os.path.exists(self.INDEX_PATH):
             raise FileNotFoundError(f"Faiss index not found: {self.INDEX_PATH}")
        self.index = faiss.read_index(self.INDEX_PATH)
        print(f"Loaded Faiss index with {self.index.ntotal} vectors.")

        if not os.path.exists(self.META_JSON):
             raise FileNotFoundError(f"Metadata JSON not found: {self.META_JSON}")
        with open(self.META_JSON, 'r', encoding='utf-8') as f:
            self.items_meta = json.load(f)
        print(f"Loaded metadata for {len(self.items_meta)} items.")
        for item in self.items_meta:
            item['image_path'] = str(
                resolve_image_path(item['image_path'], PROJECT_PATHS.data_dir)[0]
            )

        self.content_index = faiss.read_index(str(self.CONTENT_INDEX_PATH))

        if not os.path.exists(self.EMB_PATH):
            raise FileNotFoundError(f"Embeddings file not found: {self.EMB_PATH}")
        self.embeddings = np.load(self.EMB_PATH, mmap_mode='r')
        print(f"Loaded embeddings array with shape: {self.embeddings.shape}")

        counts = {
            "search index": self.index.ntotal,
            "content index": self.content_index.ntotal,
            "metadata": len(self.items_meta),
            "embeddings": self.embeddings.shape[0],
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"Artifact item-count mismatch: {counts}")
        print("Successfully loaded search data.")

    def search(self, query):
        if not self.load_success:
            print("Search system failed to load. Search unavailable.")
            return []
        try:
            with torch.no_grad():
                text = self.tokenizer([query]).to(self.DEVICE)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy().astype('float32')
            distances, indices = self.index.search(text_features, self.TOP_K)
            return [(self.items_meta[idx], float(dist)) for dist, idx in zip(distances[0], indices[0]) if 0 <= idx < len(self.items_meta)]
        except Exception as e:
            print(f"Search error: {e}")
            traceback.print_exc()
            return []

    def find_similar_items(self, item_idx: int, top_n=3):
        if not self.load_success:
            print("Search system failed to load. Similarity search unavailable.")
            return []
        if item_idx < 0 or item_idx >= len(self.embeddings):
            print(f"Error: Invalid item index {item_idx}")
            return []
        if self.index is None or self.embeddings is None:
             print("Error: Index or embeddings not loaded.")
             return []

        query_emb = self.embeddings[item_idx : item_idx+1].copy()
        faiss.normalize_L2(query_emb)
        try:
            distances, indices = self.content_index.search(query_emb, top_n + 1)
            results = []
            max_valid_index = len(self.items_meta)
            for dist, idx in zip(distances[0], indices[0]):
                if idx == item_idx: continue
                if not (0 <= idx < max_valid_index): continue
                item = self.items_meta[idx]
                results.append((item, dist))
                if len(results) >= top_n: break
            return results
        except Exception as e:
             print(f"Error finding similar items: {e}")
             traceback.print_exc()
             return []

    def get_metadata_by_asin(self, asin: str):
        if not self.load_success:
            return None
        return self.asin_to_metadata_map.get(asin)


# Comment Sentiment Map (Radar map + Keyword cloud)
class SentimentMapFrame(ctk.CTkFrame):

    def __init__(self, parent, item, reviews_data, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.item = item
        self.product_id = item.get('asin') or item.get('ASIN') or item.get('productID') or "UNKNOWN_ID"
        self.reviews_data = reviews_data

        self.configure(fg_color="transparent", corner_radius=15)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 5))
        top_bar.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            top_bar, text="Sentiment Summary", font=("Arial Rounded MT Bold", 14),
            text_color=self._apply_appearance_mode(("#000000", "#FFFFFF"))
        )
        title_label.grid(row=0, column=0, sticky="w", padx=(5,0))

        self.mode_var = tk.StringVar(value="positive")
        self.toggle_btn = ctk.CTkButton(
            top_bar, text="Negative", command=self.toggle_mode, corner_radius=10,
            font=("Arial Rounded MT Bold", 11), width=80, height=26,
            fg_color="#ADD8E6", hover_color="#87CEEB"
        )
        self.toggle_btn.grid(row=0, column=1, sticky="e", padx=(0, 5))

        self.chart_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0,5))
        self.chart_frame.grid_columnconfigure(0, weight=1)
        self.chart_frame.grid_rowconfigure(0, weight=1)

        product_review_info = self.reviews_data.get(self.product_id)

        if product_review_info and "aspects" in product_review_info:
            self.aspects = product_review_info.get("aspects", [])
            self.positive_scores = product_review_info.get("positiveScores", [])
            self.negative_scores = product_review_info.get("negativeScores", [])
            self.positive_keywords = product_review_info.get("positiveKeywords", [])
            self.negative_keywords = product_review_info.get("negativeKeywords", [])
            self.has_data = True
            if not self.aspects or (not self.positive_scores and not self.negative_scores):
                 self.has_data = False
                 print(f"Review data found for ASIN {self.product_id}, but lists are empty.")
        else:
            print(f"No valid review data found for ASIN: {self.product_id}. Sentiment section will show fallback or message.")
            self.has_data = False
            # Fallback data (optional, can be removed if no fallback desired)
            self.aspects = ["Quality", "Appearance", "Value", "Service"]
            self.positive_scores = [12, 20, 15, 6]
            self.negative_scores = [2, 1, 3, 1]
            self.positive_keywords = ["good", "beautiful", "great", "excellent"]
            self.negative_keywords = ["cheap", "bad"]
            self.has_data = True # Display fallback data if using it

        self.radar_fig = None
        self.radar_canvas = None
        self.wordcloud_label = None
        self.create_charts()

    def toggle_mode(self):
        if not self.has_data or not self.aspects: return
        if self.mode_var.get() == "positive":
            self.mode_var.set("negative")
            self.toggle_btn.configure(text="Positive", fg_color="#FFC0CB", hover_color="#FFB6C1")
        else:
            self.mode_var.set("positive")
            self.toggle_btn.configure(text="Negative", fg_color="#ADD8E6", hover_color="#87CEEB")
        self.create_charts()

    def clear_previous_charts(self):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        if self.radar_fig:
            try: plt.close(self.radar_fig)
            except Exception as e: print(f"Error closing previous radar figure: {e}")
            self.radar_fig = None
            self.radar_canvas = None
        self.wordcloud_label = None

    def create_charts(self):
        self.clear_previous_charts()
        if not self.has_data or not self.aspects:
             no_data_label = ctk.CTkLabel(self.chart_frame, text="No review data available\nfor sentiment analysis.",
                                          font=("Arial Rounded MT Bold", 12), text_color="gray", justify="center")
             self.chart_frame.grid_rowconfigure(0, weight=1)
             self.chart_frame.grid_columnconfigure(0, weight=1)
             no_data_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=20)
             self.toggle_btn.configure(state="disabled", text="N/A")
             return
        else:
             self.toggle_btn.configure(state="normal")

        mode = self.mode_var.get()
        if mode == "positive":
            scores = self.positive_scores
            keywords = self.positive_keywords
            radar_color = "pink"
            cloud_cmap = "Reds"
            cloud_title = "Positive Keywords"
        else:
            scores = self.negative_scores
            keywords = self.negative_keywords
            radar_color = "lightblue"
            cloud_cmap = "Blues"
            cloud_title = "Negative Keywords"

        self.chart_frame.grid_rowconfigure(0, weight=0)
        self.chart_frame.grid_columnconfigure(0, weight=1)

        valid_data_for_radar = bool(self.aspects and any(s > 0 for s in scores))
        if valid_data_for_radar:
             self.radar_canvas = self.create_radar_chart(self.aspects, scores, radar_color, fade=True)
             if self.radar_canvas:
                  radar_widget = self.radar_canvas.get_tk_widget()
                  radar_widget.pack(side="top", fill="x", expand=False, padx=5, pady=(0, 5))
        else:
             no_radar_label = ctk.CTkLabel(self.chart_frame, text=f"No significant {mode}\naspects found.",
                                           font=("Arial Rounded MT Bold", 11), text_color="gray")
             no_radar_label.pack(side="top", pady=10)

        if keywords:
             cloud_frame = self.create_keywords_cloud(keywords, cloud_cmap, cloud_title)
             if cloud_frame:
                 cloud_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        else:
            no_cloud_label = ctk.CTkLabel(self.chart_frame, text=f"No prominent {mode}\nkeywords found.",
                                           font=("Arial Rounded MT Bold", 11), text_color="gray")
            no_cloud_label.pack(side="top", expand=True, pady=10)

    def create_radar_chart(self, aspects, scores, color, fade=True):
        import numpy as np
        if not aspects or not scores or len(aspects) != len(scores): return None
        if len(aspects) < 3: return None
        scores = [float(s) if isinstance(s, (int, float)) and np.isfinite(s) else 0 for s in scores]
        if all(s == 0 for s in scores): return None

        angles = np.linspace(0, 2 * np.pi, len(aspects), endpoint=False).tolist()
        scores_plot = np.concatenate((scores, [scores[0]]))
        angles_plot = np.concatenate((angles, [angles[0]]))
        fig_bg_color = self._apply_appearance_mode(("#FFFFFF", "#2B2B2B"))
        try:
            if self.radar_fig: plt.close(self.radar_fig)
        except Exception: pass

        self.radar_fig = plt.Figure(figsize=(3, 3), dpi=100, facecolor=fig_bg_color)
        ax = self.radar_fig.add_subplot(111, polar=True, facecolor=fig_bg_color)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        tick_labels = [lbl[:12] + '...' if len(lbl) > 12 else lbl for lbl in aspects]
        ax.set_xticklabels(tick_labels, fontsize=8, color=self._apply_appearance_mode(("#333333", "#CCCCCC")))
        ax.tick_params(axis='x', which='major', pad=1)
        max_score = max(scores) if scores else 1
        y_limit = max(1, max_score) * 1.2
        ax.set_ylim(0, y_limit)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(color='gray', linestyle=':', linewidth=0.5)
        line, = ax.plot(angles_plot, np.zeros_like(scores_plot), color=color, linewidth=2)
        fill = ax.fill(angles_plot, np.zeros_like(scores_plot), color=color, alpha=0.25)
        canvas = FigureCanvasTkAgg(self.radar_fig, master=self.chart_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(bg=self._apply_appearance_mode(("#FFFFFF", "#2B2B2B")))
        steps = 10
        delay_ms = 50
        target_scores = scores_plot

        def animate_radar(step_i=1):
            if not canvas or not canvas.get_tk_widget().winfo_exists(): return
            fraction = step_i / float(steps)
            current_scores = target_scores * fraction
            line.set_ydata(current_scores)
            try: path = line.get_path(); patch = fill[0]; patch.set_xy(path.vertices)
            except IndexError: pass
            except Exception as e_anim: print(f"Error during radar animation update: {e_anim}"); return
            try: canvas.draw_idle()
            except tk.TclError: return
            except Exception as e_draw: print(f"Error during radar canvas draw: {e_draw}"); return
            if step_i < steps:
                self.after(delay_ms, lambda: animate_radar(step_i + 1))

        if fade: self.after(50, animate_radar)
        else:
             line.set_ydata(target_scores)
             try: path = line.get_path(); patch = fill[0]; patch.set_xy(path.vertices)
             except IndexError: pass
             except Exception as e_fill: print(f"Error setting final radar fill: {e_fill}")
             try: canvas.draw_idle()
             except tk.TclError: pass
             except Exception as e_draw_final: print(f"Error drawing final radar: {e_draw_final}")

        def on_destroy(event):
             widget_path = str(event.widget); self_path = str(self)
             if widget_path == self_path or widget_path.startswith(self_path + '.'):
                if self.radar_fig:
                    try: plt.close(self.radar_fig)
                    except Exception as e_close: print(f"Error closing radar figure on destroy: {e_close}")
                    self.radar_fig = None
        self.bind("<Destroy>", on_destroy, add="+")
        return canvas

    def create_keywords_cloud(self, keywords, cmap, title):
        frame = ctk.CTkFrame(self.chart_frame, fg_color="#FFFFFF")
        if not keywords:
             no_kw_label = ctk.CTkLabel(frame, text="No keywords found.", font=("Arial", 10), text_color="gray")
             no_kw_label.pack(expand=True, pady=20)
             return frame
        text = " ".join(keywords)
        try:
            wc = WordCloud(
                width=400, height=300, background_color="white", colormap=cmap,
                max_font_size=30, min_font_size=8, scale=2, max_words=500
            ).generate(text)
            wc_img_pil = wc.to_image()
            ctk_img = ctk.CTkImage(light_image=wc_img_pil, dark_image=wc_img_pil, size=(400, 300))
            img_label = ctk.CTkLabel(frame, text="", image=ctk_img)
            img_label.image = ctk_img
            img_label.pack(fill="both", expand=True)
        except Exception as e:
            print(f"Error generating word cloud: {e}")
            traceback.print_exc()
            error_label = ctk.CTkLabel(frame, text="Error generating cloud.", font=("Arial", 10), text_color="red")
            error_label.pack(expand=True, pady=20)
        return frame


# Main application interface
class ModernSearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.attributes("-alpha", 0.0)
        self.fade_in_step = 0.0
        self.search_system = ProductSearchSystem()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="product-search")
        self.search_future = None
        if not self.search_system.load_success:
            print("WARNING: Product Search System failed to load. Search will not function.")

        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")
        self.title("Intelligent Product Search")
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.geometry("1300x800")
        self.update_idletasks()
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        win_w, win_h = 1300, 800
        x_pos, y_pos = (screen_w // 2) - (win_w // 2), (screen_h // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x_pos}+{y_pos}")

        self.header_color = self._apply_appearance_mode(("#232F3E", "#1A242F"))
        self.content_color = self._apply_appearance_mode(("#FFFFFF", "#2B2B2B"))
        self.bg_color = self._apply_appearance_mode(("#F0F0F0", "#1F1F1F"))

        # Load Reviews Data
        self.reviews_data_path = str(PROJECT_ROOT / "reviews_data.json")
        self.reviews_data = self.load_reviews_data(self.reviews_data_path)

        # Load User Recommendation Data
        self.user_purchase_data_path = str(PROJECT_PATHS.data_dir / "Dataset_Rec.csv")
        self.category_metadata_path = str(PROJECT_PATHS.data_dir / "train_updated_final.csv")
        self.user_purchase_data = None
        self.category_metadata = None
        self.interaction_recommender = None
        self.user_rec_load_success = self.load_user_recommendation_data()

        self.available_categories = self.get_available_categories()
        self.selected_category = "All"
        self.min_price_val = 0
        self.max_price_val = 500
        self.current_results = []
        self.raw_search_results = []
        self.current_view = "recommendations"
        self.last_search_query = ""
        self.last_sort_option = "Match Score"

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.container = ctk.CTkFrame(self, fg_color=self.bg_color, corner_radius=10)
        self.container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.container.grid_rowconfigure(0, weight=0)
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_rowconfigure(2, weight=0)
        self.container.grid_columnconfigure(0, weight=1)

        self.setup_search_section()
        self.content_frame = ctk.CTkFrame(self.container, corner_radius=10, fg_color=self.content_color)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 5))
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.container, text="", font=("Arial Rounded MT Bold", 11),
            text_color=self._apply_appearance_mode(("#555555", "#AAAAAA")), anchor="w"
        )
        self.status_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 5))

        self.show_recommendations()
        self.after(50, self.fade_in_app)

    def get_available_categories(self):
        categories = set()
        if self.search_system and self.search_system.load_success and self.search_system.items_meta:
            for item in self.search_system.items_meta:
                cat = item.get("categoryName", "Unknown")
                if cat and cat != "Unknown":
                    categories.add(cat)
        else:
            print("Cannot get categories, search system metadata not available.")
            return ["All", "Others", "Shirt", "Skirt", "Socks", "Jacket",
                    "Sneakers", "Slippers", "Pants", "Shorts", "Boots", "Bike"]
        sorted_categories = sorted(list(categories))
        return ["All"] + sorted_categories

    def load_reviews_data(self, path):
        print(f"Loading reviews data from: {path}")
        if not os.path.exists(path):
            print(f"Warning: '{path}' not found. Sentiment analysis features will be limited.")
            csv_file_path = str(PROJECT_PATHS.data_dir / "Dataset_Rec.csv")
            if os.path.exists(csv_file_path):
                 print(f"Attempting to build '{path}' from '{csv_file_path}'...")
                 try:
                      build_reviews_data(csv_file_path, path)
                      if os.path.exists(path):
                           with open(path, "r", encoding="utf-8") as f: return json.load(f)
                      else: print("Failed to build reviews data."); return {}
                 except Exception as e:
                      print(f"Error building reviews data: {e}"); traceback.print_exc(); return {}
            else: print(f"Source CSV '{csv_file_path}' not found either. Cannot build reviews data."); return {}
        else:
            try:
                with open(path, "r", encoding="utf-8") as f: data = json.load(f)
                print(f"Successfully loaded reviews data for {len(data)} ASINs.")
                return data
            except Exception as e: print(f"Error loading reviews data file: {e}"); return {}

    def load_user_recommendation_data(self):
        """Loads the CSV files needed for user-based recommendations."""
        print("Loading data for user recommendations...")
        try:
            # Load Purchase Data
            if os.path.exists(self.user_purchase_data_path):
                self.user_purchase_data = pd.read_csv(self.user_purchase_data_path, on_bad_lines='skip')
                print(f"Successfully loaded user purchase data: {self.user_purchase_data_path}")
                print(f"  Columns: {self.user_purchase_data.columns.tolist()}")
                required_purchase_cols = ['ASIN', 'user']
                if not all(col in self.user_purchase_data.columns for col in required_purchase_cols):
                    print(f"  WARNING: User purchase data missing required columns: {required_purchase_cols}")
                    return False
                grouped_items = {
                    str(user): set(group['ASIN'].dropna().astype(str))
                    for user, group in self.user_purchase_data.groupby('user')
                }
                self.interaction_recommender = InteractionRecommender(grouped_items)
            else:
                print(f"Warning: User purchase data file not found: {self.user_purchase_data_path}")
                return False

            # Load Category Metadata
            if os.path.exists(self.category_metadata_path):
                 try:
                     # Try different encodings if ISO-8859-1 fails
                     self.category_metadata = pd.read_csv(self.category_metadata_path, encoding='ISO-8859-1')
                 except UnicodeDecodeError:
                     print(f"Warning: ISO-8859-1 encoding failed for {self.category_metadata_path}. Trying utf-8...")
                     try:
                         self.category_metadata = pd.read_csv(self.category_metadata_path, encoding='utf-8', on_bad_lines='skip')
                     except Exception as e_utf8:
                         print(f"Error loading category metadata with utf-8: {e_utf8}")
                         self.category_metadata = None # Failed to load

                 if self.category_metadata is not None:
                     print(f"Successfully loaded category metadata: {self.category_metadata_path}")
                     print(f"  Columns: {self.category_metadata.columns.tolist()}")
                     required_category_cols = ['asin', 'categoryName'] # Adjust if needed
                     if not all(col in self.category_metadata.columns for col in required_category_cols):
                         print(f"  WARNING: Category metadata missing required columns: {required_category_cols}")
                         self.category_metadata = None
                 else:
                     print(f"Warning: Failed to load category metadata file: {self.category_metadata_path}")
            else:
                print(f"Warning: Category metadata file not found: {self.category_metadata_path}")

            if self.user_purchase_data is None:
                 print("User recommendation data loading failed due to missing interaction data.")
                 return False

            print("User recommendation data loaded successfully.")
            return True

        except FileNotFoundError as e:
            print(f"Error loading user recommendation data: File not found - {e}")
            return False
        except pd.errors.EmptyDataError as e:
             print(f"Error loading user recommendation data: Empty file - {e}")
             return False
        except Exception as e:
            print(f"An unexpected error occurred loading user recommendation data: {e}")
            traceback.print_exc()
            return False


    def fade_in_app(self):
        if self.fade_in_step < 1.0:
            self.fade_in_step += 0.04
            self.attributes("-alpha", self.fade_in_step)
            self.after(25, self.fade_in_app)
        else:
            self.attributes("-alpha", 1.0)

    def fade_in_card(self, card, from_color="#E0E0E0", to_color=None, steps=10, delay=15, step=0):
        if not card.winfo_exists(): return
        if to_color is None:
             try:
                 to_color = card._fg_color
                 if isinstance(to_color, (list, tuple)): to_color = self._apply_appearance_mode(tuple(to_color))
             except AttributeError: to_color = self._apply_appearance_mode(("#FFFFFF", "#333333"))
        if not isinstance(from_color, str) or not from_color.startswith('#'): from_color="#E0E0E0"
        if not isinstance(to_color, str) or not to_color.startswith('#'): to_color=self._apply_appearance_mode(("#FFFFFF", "#333333"))

        fraction = step / float(steps)
        try:
            new_color = interpolate_color(from_color, to_color, fraction)
            if card.winfo_exists(): card.configure(fg_color=new_color)
        except tk.TclError: return
        except Exception as e:
            print(f"Error during fade_in_card interpolation/configure: {e}")
            try:
                if card.winfo_exists(): card.configure(fg_color=to_color)
            except tk.TclError: pass
            return

        if step < steps:
            if card.winfo_exists(): self.after(delay, lambda: self.fade_in_card(card, from_color, to_color, steps, delay, step + 1))
        elif card.winfo_exists():
             try: card.configure(fg_color=to_color)
             except tk.TclError: pass

    def setup_search_section(self):
        """ Sets up the top header section with title, search, filters. """
        header_frame = ctk.CTkFrame(self.container, corner_radius=10, fg_color=self.header_color)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(5, 2))
        title = ctk.CTkLabel(title_frame, text="Intelligent Product Search", font=("Arial Rounded MT Bold", 18), text_color="#FFFFFF")
        title.pack(pady=2)

        input_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=80, pady=(2, 4), expand=True)
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=0)

        self.search_entry = ctk.CTkEntry(
            input_frame, placeholder_text="e.g., 'comfortable running shoes for summer'",
            height=30, font=("Arial", 13), corner_radius=15,
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self.perform_search())

        self.search_button = ctk.CTkButton(
            input_frame, text="Search", command=self.perform_search, height=30, width=90,
            corner_radius=15, font=("Arial Rounded MT Bold", 12), fg_color="#FF9900", hover_color="#E08E0B"
        )
        self.search_button.grid(row=0, column=1)
        if not self.search_system or not self.search_system.load_success:
             self.search_button.configure(state="disabled", text="N/A")
             self.search_entry.configure(state="disabled", placeholder_text="Search system unavailable")

        filter_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        filter_frame.pack(fill="x", padx=100, pady=(0, 5), expand=True)
        filter_frame.grid_columnconfigure(0, weight=1)
        filter_frame.grid_columnconfigure(1, weight=0)

        slider_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        slider_frame.grid(row=0, column=0, sticky="ew", padx=(0, 15))
        slider_frame.grid_columnconfigure(0, weight=1)
        slider_frame.grid_columnconfigure(1, weight=0)

        self.range_slider = RangeSlider(
            slider_frame, total_range=(0, 500), on_change=self.on_price_change, fg_color="transparent"
        )
        self.range_slider.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        self.price_label = ctk.CTkLabel(
            slider_frame, text=f"[£{int(self.min_price_val)} - £{int(self.max_price_val)}]",
            font=("Arial Rounded MT Bold", 11), text_color="#FFFFFF", width=90
        )
        self.price_label.grid(row=0, column=1, padx=(8, 0))

        cat_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        cat_frame.grid(row=0, column=1, sticky="e")

        self.category_var = tk.StringVar(value=self.selected_category)
        self.category_optionmenu = ctk.CTkOptionMenu(
            cat_frame, values=self.available_categories, variable=self.category_var,
            command=self.on_category_change, width=150, corner_radius=10,
            font=("Arial Rounded MT Bold", 11), height=28,
        )
        self.category_optionmenu.pack(side="left")
        if not self.search_system or not self.search_system.load_success:
             self.category_optionmenu.configure(state="disabled")

        self.no_results_label = ctk.CTkLabel(
            header_frame, text="", font=("Arial Rounded MT Bold", 12), text_color="#FF9900"
        )
        self.no_results_label.pack(pady=(0, 5))

    def on_category_change(self, choice):
        print(f"Category changed to: {choice}")
        self.selected_category = choice
        if self.current_view == "recommendations":
            self.show_recommendations()
        elif self.current_view == "results":
            if hasattr(self, 'raw_search_results') and self.raw_search_results:
                 self.apply_filters_and_redisplay()

    def on_price_change(self, left_val, right_val):
        self.min_price_val = left_val
        self.max_price_val = right_val
        self.price_label.configure(text=f"[£{int(left_val)} - £{int(right_val)}]")
        if self.current_view == "recommendations":
             self._schedule_recommendation_refresh()
        elif self.current_view == "results":
             if hasattr(self, 'raw_search_results') and self.raw_search_results:
                self._schedule_filter_apply()

    _filter_job = None
    def _schedule_filter_apply(self):
        if self._filter_job: self.after_cancel(self._filter_job)
        self._filter_job = self.after(400, self.apply_filters_and_redisplay)

    _recommend_refresh_job = None
    def _schedule_recommendation_refresh(self):
        if self._recommend_refresh_job: self.after_cancel(self._recommend_refresh_job)
        self._recommend_refresh_job = self.after(400, self.show_recommendations)

    def clear_content_frame(self):
        for widget in self.content_frame.winfo_children(): widget.destroy()
        self.status_label.configure(text="")

    def show_recommendations(self):
        print("Showing recommendations view")
        self.current_view = "recommendations"
        self.clear_content_frame()
        self.status_label.configure(text="Showing popular items. Use search or filters.")
        self.no_results_label.configure(text="")
        self.current_results = []
        self.raw_search_results = []
        self.last_search_query = ""

        self.recommended_items = self.get_filtered_recommendations()

        if not self.recommended_items:
             no_rec_label = ctk.CTkLabel(self.content_frame, text="No recommendations match the current filters.",
                                           font=("Arial Rounded MT Bold", 14), text_color="gray")
             no_rec_label.pack(expand=True, padx=20, pady=20)
             return

        self.recommend_scroll = ctk.CTkScrollableFrame(self.content_frame, fg_color=self.content_color, corner_radius=25)
        self.recommend_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        self.recommend_page = 0
        self.PAGE_SIZE = 16
        self.recommend_widgets = []
        self.cols_rec = 4
        for c in range(self.cols_rec):
            self.recommend_scroll.grid_columnconfigure(c, weight=1, uniform="rec_col")
        self.load_more_recommendations()

    def get_filtered_recommendations(self):
        print(f"Filtering recommendations for Category: {self.selected_category}, Price: £{self.min_price_val:.0f}-£{self.max_price_val:.0f}")
        if not self.search_system or not self.search_system.load_success or not self.search_system.items_meta:
            print("Metadata not available for recommendations.")
            return []
        items_to_filter = self.search_system.popular_items
        if self.selected_category != "All":
            selected_cat_lower = self.selected_category.lower()
            items_to_filter = self.search_system.items_by_category.get(selected_cat_lower, [])

        def get_price_float(item):
            try: return float(item.get('price', None))
            except: return None
        price_filtered = [item for item in items_to_filter if (p := get_price_float(item)) is not None and self.min_price_val <= p <= self.max_price_val]

        print(f"Found {len(price_filtered)} recommendations after filtering.")
        return price_filtered

    def load_more_recommendations(self):
        start_idx = self.recommend_page * self.PAGE_SIZE
        end_idx = start_idx + self.PAGE_SIZE
        page_items = self.recommended_items[start_idx:end_idx]

        if not page_items:
            if self.recommend_page > 0:
                 status_text = f"Showing all {len(self.recommended_items)} matching recommendations."
                 self.status_label.configure(text=status_text)
                 if hasattr(self, 'load_more_btn') and self.load_more_btn.winfo_exists():
                     self.load_more_btn.destroy(); delattr(self, 'load_more_btn')
            elif self.recommend_page == 0 and len(self.recommended_items) == 0:
                 self.status_label.configure(text="No recommendations match the current filters.")
            return

        print(f"Loading recommendation page {self.recommend_page + 1} ({len(page_items)} items)")
        rows_on_page = (len(page_items) + self.cols_rec - 1) // self.cols_rec

        for i, item in enumerate(page_items):
            current_total_items = start_idx + i
            row = current_total_items // self.cols_rec
            col = current_total_items % self.cols_rec
            card = self.create_recommendation_card_small(self.recommend_scroll, item)
            if card:
                card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                delay = (i % (self.cols_rec * 2)) * 25
                card_bg = self._apply_appearance_mode(("#FFFFFF", "#333333"))
                self.after(delay, lambda c=card, bg=card_bg: self.fade_in_card(c, from_color="#E8E8E8", to_color=bg))

        self.recommend_page += 1
        if hasattr(self, 'load_more_btn') and self.load_more_btn.winfo_exists():
             self.load_more_btn.destroy(); delattr(self, 'load_more_btn')

        if end_idx < len(self.recommended_items):
             button_row = (start_idx // self.cols_rec) + rows_on_page
             self.load_more_btn = ctk.CTkButton(
                  self.recommend_scroll, text="Load More", command=self.load_more_recommendations,
                  width=120, height=32, corner_radius=16, font=("Arial Rounded MT Bold", 12),
                  fg_color="#FF9900", hover_color="#E08E0B"
             )
             self.load_more_btn.grid(row=button_row, column=0, columnspan=self.cols_rec, pady=(10, 10))
        else:
             status_text = f"Showing all {len(self.recommended_items)} matching recommendations."
             self.status_label.configure(text=status_text)

    def create_recommendation_card_small(self, parent, item):
        """ Creates a smaller card for the recommendation grid view. """
        card_bg = self._apply_appearance_mode(("#FFFFFF", "#333333"))
        card = ctk.CTkFrame(parent, corner_radius=8, fg_color=card_bg)
        card.grid_rowconfigure(1, weight=0)
        card.grid_columnconfigure(0, weight=1)

        img_size = 120
        img_frame = ctk.CTkFrame(card, fg_color="transparent", corner_radius=6, width=img_size, height=img_size)
        img_frame.grid(row=0, column=0, sticky="nwe", padx=8, pady=(8, 4))
        img_frame.pack_propagate(False); img_frame.grid_propagate(False)
        img_frame.grid_rowconfigure(0, weight=1); img_frame.grid_columnconfigure(0, weight=1)

        try:
            img_path = item.get('image_path')
            if img_path and os.path.exists(img_path):
                pil_img = Image.open(img_path).convert('RGB')
                pil_img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(pil_img.width, pil_img.height))
                img_label = ctk.CTkLabel(img_frame, image=ctk_img, text="")
                img_label.grid(row=0, column=0, sticky="nsew")
            else: placeholder = ctk.CTkLabel(img_frame, text="No Image", font=("Arial", 10), text_color="gray"); placeholder.grid(row=0, column=0, sticky="nsew")
        except Exception as e:
            print(f"Error loading image for rec card {item.get('asin')}: {e}")
            placeholder = ctk.CTkLabel(img_frame, text="Img Error", font=("Arial", 10), text_color="red"); placeholder.grid(row=0, column=0, sticky="nsew")

        text_frame = ctk.CTkFrame(card, fg_color="transparent")
        text_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        text_frame.grid_columnconfigure(0, weight=1)

        title_str = item.get("title", "No Title Available"); max_title_len = 45
        display_title = (title_str[:max_title_len] + '...') if len(title_str) > max_title_len else title_str
        title_label = ctk.CTkLabel(text_frame, text=display_title, font=("Arial", 11), wraplength=150, anchor="nw", justify="left")
        title_label.grid(row=0, column=0, sticky="ew", pady=(0, 2))

        price_rating_frame = ctk.CTkFrame(text_frame, fg_color="transparent")
        price_rating_frame.grid(row=1, column=0, sticky="ew"); price_rating_frame.grid_columnconfigure(0, weight=1); price_rating_frame.grid_columnconfigure(1, weight=0)
        price_text = f"£{item.get('price','N/A')}"
        price_label = ctk.CTkLabel(price_rating_frame, text=price_text, font=("Arial Rounded MT Bold", 11), text_color="#B12704", anchor="w")
        price_label.grid(row=0, column=0, sticky="w")
        try:
            stars_val = float(item.get('stars', 0))
            if stars_val > 0:
                stars_str = f"{stars_val:.1f} ★"
                rating_label = ctk.CTkLabel(price_rating_frame, text=stars_str, font=("Arial", 10), text_color="#FFA500")
                rating_label.grid(row=0, column=1, sticky="e", padx=(5,0))
        except ValueError: pass

        details_btn = ctk.CTkButton(
            card, text="Details", command=lambda it=item: self.show_product_details(it),
            height=24, corner_radius=6, font=("Arial Rounded MT Bold", 10),
            fg_color="#FF9900", hover_color="#E08E0B", border_spacing=2
        )
        details_btn.grid(row=2, column=0, pady=(0, 8), padx=8)
        return card

    def show_search_results(self):
        print("Showing search results view")
        self.current_view = "results"
        self.clear_content_frame()
        self.no_results_label.configure(text="")

        results_frame = ctk.CTkFrame(self.content_frame, corner_radius=25, fg_color=self.content_color)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        top_bar = ctk.CTkFrame(results_frame, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=10)

        return_button = ctk.CTkButton(
            top_bar, text="Return", command=self.return_to_main, height=40, width=80,
            corner_radius=15, font=("Arial Rounded MT Bold", 13), fg_color="#FFA41C"
        )
        return_button.pack(side="left", padx=(0, 20))

        sort_options = ["Match Score", "Price (Ascending)", "Price (Descending)", "Rating"]
        if self.last_sort_option not in sort_options: self.last_sort_option = sort_options[0]
        self.sort_var = tk.StringVar(value=self.last_sort_option)
        sort_dropdown = ctk.CTkOptionMenu(
            top_bar, values=sort_options, variable=self.sort_var, command=self.sort_results,
            width=150, corner_radius=15, font=("Arial Rounded MT Bold", 13)
        )
        sort_dropdown.pack(side="left")

        self.results_container = ctk.CTkScrollableFrame(results_frame, corner_radius=25, fg_color="#FCFCFC")
        self.results_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.results_container.grid_columnconfigure(0, weight=1)

    def display_results_list(self):
         """ Populates the results_container with result cards. """
         for widget in self.results_container.winfo_children(): widget.destroy()
         if not self.current_results:
              no_res_label = ctk.CTkLabel(self.results_container, text="No matching products found.",
                                            font=("Arial Rounded MT Bold", 14), text_color="gray")
              no_res_label.pack(expand=True, padx=20, pady=40)
              self.no_results_label.configure(text="No products found matching your query and filters.", text_color="#FF9900")
              self.status_label.configure(text="")
              return

         status_color = self._apply_appearance_mode(("#006400", "#50C878"))
         self.status_label.configure(
             text=f"Found {len(self.current_results)} products matching '{self.last_search_query}' [£{int(self.min_price_val)}-£{int(self.max_price_val)}, {self.selected_category}]",
             text_color=status_color
         )
         self.no_results_label.configure(text="")
         for i, (item, score) in enumerate(self.current_results): self.show_result(item, score, i)

    def show_result(self, item, distance, rank):
        """ Creates and displays a single search result item card. """
        card_bg = self._apply_appearance_mode(("#FFFFFF", "#333333"))
        card = ctk.CTkFrame(self.results_container, corner_radius=25, fg_color="#DDDDDD")
        card.pack(fill="x", padx=10, pady=10)
        self.after(rank * 80, lambda c=card, bg=card_bg: self.fade_in_card(c, from_color="#DDDDDD", to_color=bg, steps=15, delay=20))

        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="x", padx=10, pady=10)

        try:
            img_path = item.get('image_path')
            if img_path and os.path.exists(img_path):
                pil_img = Image.open(img_path).convert('RGB')
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(100, 100))
                img_label = ctk.CTkLabel(content_frame, image=ctk_img, text="")
                img_label.image = ctk_img
                img_label.pack(side="left", padx=(0, 10))
        except Exception as e: print(f"Error loading image for result {item.get('asin')}: {e}")

        text_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        title_label = ctk.CTkLabel(
            text_frame, text=item.get('title', 'No Title Available'), font=("Arial Rounded MT Bold", 14),
            wraplength=600, anchor="w", justify="left", text_color=self._apply_appearance_mode(("#000000", "#FFFFFF"))
        )
        title_label.pack(anchor="nw", pady=(0, 5))
        price_label = ctk.CTkLabel(text_frame, text=f"£{item.get('price', 'N/A')}", font=("Arial Rounded MT Bold", 14), text_color="#B12704")
        price_label.pack(anchor="nw", pady=(0, 4))

        rating_bestseller_frame = ctk.CTkFrame(text_frame, fg_color="transparent")
        rating_bestseller_frame.pack(anchor="nw", pady=(0, 5))
        stars_val = 0 # Default
        try:
            stars_val = float(item.get('stars', 0))
            if stars_val > 0:
                stars_str = "★" * int(round(stars_val))
                rating_text = f"{stars_str} ({stars_val:.1f})"
                rating_label = ctk.CTkLabel(rating_bestseller_frame, text=rating_text, font=("Arial Rounded MT Bold", 13), text_color="#FFD700")
                rating_label.pack(side="left", anchor="w")
        except ValueError: pass
        if item.get('isBestSeller', False):
             best_seller_label = ctk.CTkLabel(rating_bestseller_frame, text="✨ Best Seller", font=("Arial Rounded MT Bold", 12), text_color="#FFD700")
             pad_x = (10, 0) if stars_val > 0 else (0, 0)
             best_seller_label.pack(side="left", anchor="w", padx=pad_x)

        right_panel = ctk.CTkFrame(content_frame, fg_color="transparent")
        right_panel.pack(side="right", fill="y", padx=(10, 0))
        match_score = similarity_percent(distance)
        if match_score >= 75: score_color = "#50C878"
        elif match_score >= 50: score_color = "#FFD700"
        else: score_color = "#DC3545"
        score_frame = ctk.CTkFrame(right_panel, corner_radius=15, fg_color=score_color)
        score_frame.pack(pady=(0, 10))
        score_label = ctk.CTkLabel(score_frame, text=f"{match_score:.1f}%", font=("Arial Rounded MT Bold", 12), text_color="black" if match_score >= 50 else "white")
        score_label.pack(padx=8, pady=4)

        details_button = ctk.CTkButton(
            right_panel, text="View Details", command=lambda it=item: self.show_product_details(it),
            height=30, width=100, corner_radius=15, font=("Arial Rounded MT Bold", 12), fg_color="#FF9900"
        )
        details_button.pack(pady=(0, 5))

        product_url = item.get('productURL')
        if product_url and isinstance(product_url, str) and product_url.startswith('http'):
            link_button = ctk.CTkButton(
                right_panel, text="Product Link", command=lambda url=product_url: self.open_url(url),
                height=30, width=100, corner_radius=15, font=("Arial Rounded MT Bold", 12), fg_color="#FFA41C"
            )
            link_button.pack()

    # ---- Helper Function for User Rec ----
    def _get_item_metadata_by_asin(self, asin: str):
        """Gets item metadata dictionary from the main list using ASIN."""
        if self.search_system and self.search_system.load_success:
            return self.search_system.get_metadata_by_asin(asin)
        return None

    # ---- Helper Function for User Rec ----
    def _recommend_based_on_category_helper(self, item_id: str, purchase_data: pd.DataFrame):

        if purchase_data is None or item_id is None:
            return None
        try:
            target_item_entries = purchase_data[purchase_data['ASIN'] == item_id]
            if not target_item_entries.empty:
                # Assuming 'Production' column holds the category name
                target_item_category = target_item_entries['Production'].iloc[0]
                print(f"Target item '{item_id}' category found: {target_item_category}")
                return target_item_category
            else:
                print(f"Warning: Could not find item '{item_id}' in purchase data.")
                return None
        except KeyError as e:
            print(f"Error accessing columns in purchase data: {e}. Check column names ('ASIN', 'Production').")
            return None
        except Exception as e:
            print(f"Error finding target category for '{item_id}': {e}")
            return None

    # ---- Helper Function for User Rec ----
    def _recommend_ranked_items_from_category_helper(self, category: str, category_meta_df: pd.DataFrame, exclude_asin: str = None, top_n=3):
        """
        Finds the highest-rated items from a category as a deterministic fallback.
        Returns a list of ASINs.
        """
        recommended_items = []
        if category is None:
            return recommended_items
        try:
            if category_meta_df is not None:
                same_category_items_df = category_meta_df[
                    category_meta_df['categoryName'] == category
                ]
                all_asins_in_category = same_category_items_df['asin'].tolist()
            else:
                catalog_items = self.search_system.items_by_category.get(
                    str(category).casefold(), []
                )
                all_asins_in_category = [
                    metadata.get('asin') or metadata.get('ASIN')
                    for metadata in catalog_items
                ]
                all_asins_in_category = [asin for asin in all_asins_in_category if asin]

            if not all_asins_in_category:
                 print(f"No items found in category metadata for category: {category}")
                 return recommended_items

            # Exclude the currently viewed item if specified
            if exclude_asin:
                all_asins_in_category = [asin for asin in all_asins_in_category if asin != exclude_asin]

            if not all_asins_in_category:
                 print(f"No other items found in category '{category}' after exclusion.")
                 return recommended_items

            ranked_items = []
            for asin in all_asins_in_category:
                metadata = self._get_item_metadata_by_asin(asin) or {}
                try:
                    rating = float(metadata.get('stars', 0))
                except (TypeError, ValueError):
                    rating = 0.0
                try:
                    review_count = int(float(metadata.get('reviews', 0)))
                except (TypeError, ValueError):
                    review_count = 0
                ranked_items.append((asin, rating, review_count))
            ranked_items.sort(key=lambda row: (-row[1], -row[2], row[0]))
            recommended_items = [asin for asin, _, _ in ranked_items[:top_n]]
            print(f"Recommended ASINs from category '{category}': {recommended_items}")

        except KeyError as e:
             print(f"Error accessing columns in category metadata: {e}. Check column names ('categoryName', 'asin').")
        except Exception as e:
            print(f"Error getting category recommendations for '{category}': {e}")

        return recommended_items


    def show_product_details(self, item):
        print(f"Showing details for: {item.get('asin', 'Unknown ASIN')}")
        self.current_view = "details"
        self.clear_content_frame()
        self.no_results_label.configure(text="")

        detail_scroll = ctk.CTkScrollableFrame(self.content_frame, corner_radius=25, fg_color=self.content_color)
        detail_scroll.pack(fill="both", expand=True, padx=20, pady=20)

        top_bar = ctk.CTkFrame(detail_scroll, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=(10, 0))
        back_button = ctk.CTkButton(
            top_bar, text="Return", command=self.return_to_previous_view,
            height=40, width=80, corner_radius=15, font=("Arial Rounded MT Bold", 13), fg_color="#FFA41C"
        )
        back_button.pack(side="left", padx=(0, 20))

        main_frame = ctk.CTkFrame(detail_scroll, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=2) # Product info wider
        main_frame.grid_columnconfigure(1, weight=1) # Sentiment narrower

        product_card_bg = self._apply_appearance_mode(("#FCFCFC", "#333333"))
        product_card = ctk.CTkFrame(main_frame, corner_radius=25, fg_color="#DDDDDD")
        product_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        self.fade_in_card(product_card, "#DDDDDD", product_card_bg, steps=15, delay=20)
        product_card.grid_rowconfigure(0, weight=1)
        product_card.grid_columnconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(product_card, fg_color="transparent")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        try:
            img_path = item.get('image_path')
            if img_path and os.path.exists(img_path):
                pil_img = Image.open(img_path).convert('RGB')
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(300, 300))
                img_label = ctk.CTkLabel(left_panel, image=ctk_img, text="")
                img_label.image = ctk_img; img_label.pack(padx=10, pady=10)
            else: placeholder = ctk.CTkLabel(left_panel, text="No Image", height=10, text_color="#000000"); placeholder.pack(pady=(20, 5))
        except Exception as e: print(f"Error detail image: {e}"); placeholder = ctk.CTkLabel(left_panel, text="No Image", height=10, text_color="#000000"); placeholder.pack(pady=(20, 5))

        # Text Details
        title_label = ctk.CTkLabel(left_panel, text=item.get('title', 'No Title'), font=("Arial Rounded MT Bold", 18), wraplength=400, anchor="w", text_color="#000000")
        title_label.pack(anchor="nw", pady=(0, 10))
        price_label = ctk.CTkLabel(left_panel, text=f"Price: £{item.get('price','N/A')}", font=("Arial Rounded MT Bold", 14), text_color="#B12704")
        price_label.pack(anchor="nw")
        category_str = item.get('categoryName', 'Unknown')
        category_label = ctk.CTkLabel(left_panel, text=f"Category: {category_str}", font=("Arial Rounded MT Bold", 14), text_color="#000000")
        category_label.pack(anchor="nw", pady=(0, 10))
        try:
            stars_val = float(item.get('stars', 0))
            if stars_val > 0:
                stars_str = "★" * int(round(stars_val)); rating_text = f"Rating: {stars_str} ({stars_val:.1f})"
                rating_label = ctk.CTkLabel(left_panel, text=rating_text, font=("Arial Rounded MT Bold", 13), text_color="#FFD700")
                rating_label.pack(anchor="nw")
        except ValueError: pass
        product_url = item.get('productURL')
        if product_url and isinstance(product_url, str) and product_url.startswith('http'):
            link_button = ctk.CTkButton(left_panel, text="Open in Browser", command=lambda url=product_url: self.open_url(url), height=30, corner_radius=15, font=("Arial Rounded MT Bold", 12), fg_color="#FFA41C")
            link_button.pack(anchor="nw", pady=5)

        # Right: Sentiment Card
        sentiment_card_bg = self._apply_appearance_mode(("#F0F0F0", "#2A2A2A"))
        sentiment_card = ctk.CTkFrame(main_frame, corner_radius=25, fg_color="#DDDDDD")
        sentiment_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        self.fade_in_card(sentiment_card, "#DDDDDD", sentiment_card_bg, steps=15, delay=20)
        SentimentMapFrame(sentiment_card, item, self.reviews_data, fg_color="transparent").pack(fill="both", expand=True, padx=10, pady=10)

        # Bottom: SIMILARITY Recommendations (CLIP based)
        bottom_card = ctk.CTkFrame(detail_scroll, corner_radius=25, fg_color=self.content_color)
        bottom_card.pack(fill="x", expand=False, padx=10, pady=(10, 10))

        recom_label = ctk.CTkLabel(bottom_card, text="Similar Products (Content-Based)", font=("Arial Rounded MT Bold", 16), text_color="#000000")
        recom_label.pack(pady=10)
        grid_frame = ctk.CTkFrame(bottom_card, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=10, pady=10)
        rows, cols = 1, 3
        for c in range(cols): grid_frame.grid_columnconfigure(c, weight=1, uniform="rec_detail")

        similar_items = []
        try:
             if self.search_system and self.search_system.load_success:
                 current_item_asin = item.get('asin') or item.get('ASIN')
                 idx = self.search_system.asin_to_index_map.get(current_item_asin)
                 if idx is not None:
                     raw_similar_items = self.search_system.find_similar_items(idx, top_n=10)
                     detail_item_category = item.get('categoryName', 'Unknown')
                     filtered_similar = [(rec_item, dist_val) for rec_item, dist_val in raw_similar_items if rec_item.get('categoryName', 'Unknown') == detail_item_category]
                     similar_items = filtered_similar[:3]
                 else: print(f"Warning: Could not find index for ASIN {current_item_asin}.")
             else: print("Search system unavailable for similar items.")
        except Exception as e: print(f"Error fetching similar items: {e}"); traceback.print_exc()

        for i in range(cols):
            if i < len(similar_items):
                rec_item, dist_val = similar_items[i]
                rec_card = self.create_recommendation_card(grid_frame, rec_item)
                rec_card.grid(row=0, column=i, padx=15, pady=15, sticky="nsew")
            else:
                no_rec_card = ctk.CTkFrame(grid_frame, corner_radius=25, fg_color="#FFFFFF")
                no_rec_card.grid(row=0, column=i, padx=15, pady=15, sticky="nsew")
                no_rec_label = ctk.CTkLabel(no_rec_card, text="No Similar Item", font=("Arial Rounded MT Bold", 13), text_color="#000000")
                no_rec_label.pack(expand=True)

        # USER Recommendations (Category Based)
        user_rec_card = ctk.CTkFrame(detail_scroll, corner_radius=25, fg_color=self.content_color)
        user_rec_card.pack(fill="x", expand=False, padx=10, pady=(0, 20))

        user_rec_label = ctk.CTkLabel(user_rec_card, text="Customers Also Considered", font=("Arial Rounded MT Bold", 16), text_color="#000000")
        user_rec_label.pack(pady=10)
        user_rec_grid_frame = ctk.CTkFrame(user_rec_card, fg_color="transparent")
        user_rec_grid_frame.pack(fill="both", expand=True, padx=10, pady=10)
        user_rec_cols = 3
        for c in range(user_rec_cols): user_rec_grid_frame.grid_columnconfigure(c, weight=1, uniform="user_rec_detail")

        user_recommended_asins = []
        if self.user_rec_load_success and self.interaction_recommender:
            current_item_asin = item.get('asin') or item.get('ASIN')
            user_recommended_asins = [
                asin for asin, _ in self.interaction_recommender.recommend(
                    current_item_asin, top_n=user_rec_cols
                )
            ]
            if len(user_recommended_asins) < user_rec_cols:
                target_category = item.get('categoryName')
                fallback = self._recommend_ranked_items_from_category_helper(
                    target_category,
                    self.category_metadata,
                    exclude_asin=current_item_asin,
                    top_n=user_rec_cols,
                )
                user_recommended_asins.extend(
                    asin for asin in fallback if asin not in user_recommended_asins
                )
                user_recommended_asins = user_recommended_asins[:user_rec_cols]
        else:
            print("User recommendation data not loaded. Skipping user recommendations.")

        # Fetch metadata for the recommended ASINs
        user_recommended_items_meta = []
        if user_recommended_asins:
            for rec_asin in user_recommended_asins:
                meta = self._get_item_metadata_by_asin(rec_asin)
                if meta:
                    user_recommended_items_meta.append(meta)
                else:
                    print(f"Warning: Metadata not found for recommended ASIN: {rec_asin}")

        # Display user recommendations
        for i in range(user_rec_cols):
            if i < len(user_recommended_items_meta):
                rec_item_meta = user_recommended_items_meta[i]
                rec_card = self.create_recommendation_card(user_rec_grid_frame, rec_item_meta)
                rec_card.grid(row=0, column=i, padx=15, pady=15, sticky="nsew")
            else:
                # Show placeholder if not enough recommendations
                no_rec_card = ctk.CTkFrame(user_rec_grid_frame, corner_radius=25, fg_color="#FFFFFF")
                no_rec_card.grid(row=0, column=i, padx=15, pady=15, sticky="nsew")
                no_rec_label = ctk.CTkLabel(no_rec_card, text="No User Item", font=("Arial Rounded MT Bold", 13), text_color="#000000")
                no_rec_label.pack(expand=True)

    def create_recommendation_card(self, parent, item, is_placeholder=False):
        card = ctk.CTkFrame(parent, corner_radius=25, fg_color="#FFFFFF")
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        if not is_placeholder and item:
            try:
                img_path = item.get('image_path')
                if img_path and os.path.exists(img_path):
                    pil_img = Image.open(img_path).convert('RGB')
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(200, 200))
                    img_label = ctk.CTkLabel(content_frame, image=ctk_img, text="")
                    img_label.image = ctk_img; img_label.pack(padx=10, pady=(10, 5))
                else: placeholder = ctk.CTkLabel(content_frame, text="No Image", text_color="#000000", height=200); placeholder.pack(pady=(20, 5))
            except Exception as e: print("Error loading image for rec card:", e); placeholder = ctk.CTkLabel(content_frame, text="No Image", text_color="#000000", height=200); placeholder.pack(pady=(20, 5))

            title_str = item.get("title", "No Title")
            title_label = ctk.CTkLabel(content_frame, text=title_str, font=("Arial Rounded MT Bold", 13), wraplength=180, text_color="#000000")
            title_label.pack(padx=10, pady=(0, 10), fill="x")

            # Add price if available
            price_text = f"£{item.get('price','N/A')}"
            price_label = ctk.CTkLabel(content_frame, text=price_text, font=("Arial Rounded MT Bold", 11), text_color="#B12704")
            price_label.pack(pady=(0, 5))

            view_btn = ctk.CTkButton(
                content_frame, text="View Details", command=lambda it=item: self.show_product_details(it),
                height=30, width=100, corner_radius=15, font=("Arial Rounded MT Bold", 12), fg_color="#FF9900"
            )
            view_btn.pack(pady=(0, 10))
        else:
            ph_label = ctk.CTkLabel(content_frame, text=item.get("title", "Placeholder") if item else "Not Available", font=("Arial Rounded MT Bold", 13), text_color="#000000")
            ph_label.pack(expand=True, pady=20)
        return card

    def return_to_previous_view(self):
        """ Returns to the search results view if results exist, otherwise recommendations. """
        if hasattr(self, 'current_results') and self.current_results:
            print("Returning to previous view (Search Results)")
            self.show_search_results()
            self.sort_results(self.last_sort_option) # Re-sort and display
        else:
            print("No previous results found, returning to main view.")
            self.return_to_main()

    def return_to_main(self):
        """ Returns to the main recommendations view. """
        print("Returning to main view (Recommendations)")
        self.current_view = "recommendations"
        self.status_label.configure(text="")
        self.no_results_label.configure(text="")
        self.current_results = []
        self.raw_search_results = []
        self.last_search_query = ""
        self.show_recommendations()

    def open_url(self, url):
        print(f"Opening URL: {url}")
        try: webbrowser.open(url, new=2)
        except Exception as e: print(f"Error opening URL {url}: {e}"); self.status_label.configure(text=f"Error opening link: {e}", text_color="red")

    def close_app(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def perform_search(self, event=None):
        if not self.search_system or not self.search_system.load_success:
             self.status_label.configure(text="Search system is unavailable.", text_color="orange"); return
        query = self.search_entry.get().strip()
        if not query: self.no_results_label.configure(text="Please enter a search query.", text_color="#FF9900"); return
        if self.search_future and not self.search_future.done():
            self.status_label.configure(text="A search is already running...", text_color="orange")
            return

        self.last_search_query = query
        print(f"Performing search for: '{query}'")
        searching_color = self._apply_appearance_mode(("#191970", "#ADD8E6"))
        self.status_label.configure(text=f"Searching for '{query}'...", text_color=searching_color)
        self.no_results_label.configure(text="")
        self.search_button.configure(state="disabled", text="Searching...")
        self.search_future = self.executor.submit(self.search_system.search, query)
        self.after(50, self._poll_search_result)

    def _poll_search_result(self):
        if not self.search_future or not self.search_future.done():
            self.after(50, self._poll_search_result)
            return
        self.search_button.configure(state="normal", text="Search")
        try:
            self.raw_search_results = self.search_future.result()
            self.apply_filters_and_redisplay()
        except Exception as e:
            print(f"An error occurred during search execution: {e}"); traceback.print_exc()
            self.status_label.configure(text="An error occurred during search.", text_color="red")
            self.no_results_label.configure(text="Error processing search.", text_color="red")
            self.current_results = []; self.raw_search_results = []
            self.show_search_results(); self.display_results_list() # Show empty results view on error

    def apply_filters_and_redisplay(self):
        """Applies current filters (price, category) to raw_search_results and updates display."""
        print(f"Applying filters: Price [£{self.min_price_val:.0f}-£{self.max_price_val:.0f}], Category [{self.selected_category}]")
        if not hasattr(self, 'raw_search_results') or self.raw_search_results is None:
            print("No raw search results to filter.")
            if self.current_view == "results":
                 self.current_results = []
                 if hasattr(self, 'results_container') and self.results_container.winfo_exists(): self.display_results_list()
                 else: self.show_search_results(); self.display_results_list()
            return

        min_p, max_p = self.min_price_val, self.max_price_val
        selected_cat_lower = self.selected_category.lower()
        filtered_results = []
        if self.raw_search_results:
            for (item, score) in self.raw_search_results:
                try: price_val = float(item.get('price', -1.0))
                except: price_val = -1.0
                price_match = (price_val != -1.0 and min_p <= price_val <= max_p)
                item_cat_lower = item.get('categoryName', "Unknown").lower()
                category_match = (selected_cat_lower == "all" or item_cat_lower == selected_cat_lower)
                if price_match and category_match: filtered_results.append((item, score))

        self.current_results = filtered_results
        self.last_sort_option = self.sort_var.get() if hasattr(self, 'sort_var') and self.sort_var.get() else "Match Score"
        if self.current_view != "results" or not hasattr(self, 'results_container') or not self.results_container.winfo_exists():
             self.show_search_results() # This creates results_container
        self.sort_results(self.last_sort_option)

    # CORRECTED sort_results method
    def sort_results(self, sort_type):
        """ Sorts the current_results list based on the selected criteria. """
        print(f"Sorting results by: {sort_type}")
        self.last_sort_option = sort_type
        if not hasattr(self, 'current_results') or not self.current_results:
            print("No results available to sort.")
            if hasattr(self, 'results_container') and self.results_container.winfo_exists():
                self.display_results_list()
            return

        if sort_type == "Match Score":
            self.current_results.sort(
                key=lambda x: x[1] if isinstance(x[1], (int, float)) else float('-inf'),
                reverse=True,
            )
        elif sort_type == "Price (Ascending)":
            def get_price_asc(t):
                try:
                    return float(t[0].get('price', float('inf')))
                except (ValueError, TypeError):
                    return float('inf')
            self.current_results.sort(key=get_price_asc)

        elif sort_type == "Price (Descending)":
            # Define helper function correctly (multi-line)
            def get_price_desc(t):
                try:
                    return float(t[0].get('price', -1.0))
                except (ValueError, TypeError):
                    return -1.0
            self.current_results.sort(key=get_price_desc, reverse=True)

        elif sort_type == "Rating":
            def get_rating(t):
                try:
                    return float(t[0].get('stars', 0.0))
                except (ValueError, TypeError):
                    return 0.0
            self.current_results.sort(key=get_rating, reverse=True)

        # After sorting, refresh the display if the container exists
        if hasattr(self, 'results_container') and self.results_container.winfo_exists():
            self.display_results_list()
        else:
            print("Warning: Results container not found, cannot display sorted results.")


def main():
    search_dir = os.environ.get('PRODUCT_SEARCH_DIR', str(PROJECT_ROOT / 'Search'))
    if not os.path.isdir(search_dir): print(f"Error: Search directory not found: {search_dir}"); return

    reviews_json_path = str(PROJECT_ROOT / "reviews_data.json")
    csv_file_path = str(PROJECT_PATHS.data_dir / "Dataset_Rec.csv")
    if not os.path.exists(reviews_json_path):
        print(f"'{reviews_json_path}' not found.")
        if os.path.exists(csv_file_path):
            print("Attempting to build reviews data from CSV...")
            try:
                build_reviews_data(csv_file_path, reviews_json_path)
                if not os.path.exists(reviews_json_path): print("Failed to build reviews data. Sentiment features might be unavailable.")
            except Exception as e: print(f"Error building reviews data: {e}"); traceback.print_exc()
        else: print(f"Source CSV '{csv_file_path}' also not found. Cannot build reviews data.")

    app = None
    try:
        app = ModernSearchApp()
        if app.winfo_exists():
            disclaimer = DisclaimerWindow(app)
            app.wait_window(disclaimer)
            if app.winfo_exists():
                 app.mainloop()
            else: print("Main application window closed before starting main loop.")
        else: print("Failed to create main application window.")
    except Exception as e:
        print("\n--- An unexpected error occurred during app initialization or runtime ---")
        print(f"Error Type: {type(e).__name__}"); print(f"Error Message: {e}"); print("Traceback:"); traceback.print_exc()
        try:
            root_tk = tk.Tk(); root_tk.withdraw()
            import tkinter.messagebox
            tkinter.messagebox.showerror("Application Error", f"A critical error occurred:\n{e}\n\nPlease check the console for details.")
            root_tk.destroy()
        except Exception as diag_e: print(f"Could not display error dialog: {diag_e}")
        # Ensure app window is destroyed if it exists after an error
        if app and app.winfo_exists(): app.destroy()

if __name__ == "__main__":
    main()
