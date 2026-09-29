"""Reclassify products with real image-and-title CLIP zero-shot scoring.

The previous implementation extracted an image vector and then discarded it, passing
only the phrase "image available" to Flan-T5. This version compares both image and
title embeddings against the same fixed label prompts and writes output incrementally.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", PROJECT_ROOT / "dataset")).resolve()
INPUT_FILE = Path(os.environ.get("FYP_RECLASSIFY_INPUT", DATA_DIR / "train.csv"))
OUTPUT_FILE = Path(
    os.environ.get("FYP_RECLASSIFY_OUTPUT", DATA_DIR / "train_updated.csv")
)
COUNTS_FILE = Path(
    os.environ.get("FYP_RECLASSIFY_COUNTS", DATA_DIR / "trainCounts.csv")
)
BATCH_SIZE = int(os.environ.get("FYP_RECLASSIFY_BATCH_SIZE", "64"))
CONFIDENCE_THRESHOLD = float(os.environ.get("FYP_RECLASSIFY_THRESHOLD", "0.35"))

CATEGORIES = [
    "Shirt",
    "Jacket",
    "Bike",
    "Pants",
    "Shorts",
    "Skirt",
    "Socks",
    "Boots",
    "Sneakers",
    "Slippers",
]
LABEL_PROMPTS = [f"a product photo and description of {label.lower()}" for label in CATEGORIES]


def resolve_image_path(raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return DATA_DIR / path


def load_images(paths: list[Path | None]) -> tuple[list[Image.Image], torch.Tensor]:
    images: list[Image.Image] = []
    available: list[bool] = []
    for path in paths:
        try:
            if path is None:
                raise FileNotFoundError
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
            available.append(True)
        except (FileNotFoundError, OSError):
            images.append(Image.new("RGB", (224, 224), color="white"))
            available.append(False)
    return images, torch.tensor(available, dtype=torch.bool)


def normalized(features: torch.Tensor) -> torch.Tensor:
    return functional.normalize(features, p=2, dim=-1)


def classify_chunk(
    dataframe: pd.DataFrame,
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    label_features: torch.Tensor,
    device: torch.device,
) -> pd.DataFrame:
    predictions: list[str] = []
    confidences: list[float] = []

    for start in range(0, len(dataframe), BATCH_SIZE):
        batch = dataframe.iloc[start : start + BATCH_SIZE]
        paths = [resolve_image_path(value) for value in batch["imgUrl"]]
        images, image_available = load_images(paths)
        titles = batch["title"].fillna("").astype(str).tolist()

        image_inputs = processor(images=images, return_tensors="pt").to(device)
        title_inputs = processor(
            text=titles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        with torch.inference_mode():
            image_features = normalized(model.get_image_features(**image_inputs))
            title_features = normalized(model.get_text_features(**title_inputs))
            image_scores = image_features @ label_features.T
            title_scores = title_features @ label_features.T

            availability = image_available.to(device).unsqueeze(1)
            combined_scores = torch.where(
                availability,
                0.4 * image_scores + 0.6 * title_scores,
                title_scores,
            )
            probabilities = functional.softmax(combined_scores * 100.0, dim=1)
            confidence, category_index = probabilities.max(dim=1)

        for index, score in zip(category_index.cpu().tolist(), confidence.cpu().tolist()):
            predictions.append(CATEGORIES[index] if score >= CONFIDENCE_THRESHOLD else "Others")
            confidences.append(float(score))

    result = dataframe.copy()
    result["categoryName"] = predictions
    result["categoryConfidence"] = confidences
    return result


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    label_inputs = processor(text=LABEL_PROMPTS, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        label_features = normalized(model.get_text_features(**label_inputs))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    category_counts: Counter[str] = Counter()
    first_chunk = True
    chunks = pd.read_csv(INPUT_FILE, chunksize=max(BATCH_SIZE * 10, 1000))
    for chunk in tqdm(chunks, desc="Classifying dataset"):
        classified = classify_chunk(
            chunk,
            model=model,
            processor=processor,
            label_features=label_features,
            device=device,
        )
        category_counts.update(classified["categoryName"])
        classified.to_csv(
            temporary_output,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8",
        )
        first_chunk = False

    temporary_output.replace(OUTPUT_FILE)
    pd.DataFrame(
        sorted(category_counts.items(), key=lambda pair: (-pair[1], pair[0])),
        columns=["Category Name", "Sample Count"],
    ).to_csv(COUNTS_FILE, index=False, encoding="utf-8")
    print(f"Saved reclassified data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
