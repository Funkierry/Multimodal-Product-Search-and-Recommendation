import pandas as pd
import os
import requests
from tqdm import tqdm
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", PROJECT_ROOT / "dataset")).resolve()

def create_directories():
    """Create necessary directories for storing data and images"""
    dirs = [DATA_DIR / 'images' / name for name in ('train', 'test', 'val')]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def download_image(url, save_path):
    """Download image from URL and save to specified path"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except:
        return False

def process_dataset(input_file=None):
    input_file = input_file or DATA_DIR / 'amz_uk_processed_data.csv'
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv(input_file)

    # Count products per category
    category_counts = Counter(df['categoryName'])

    # Get top 20 categories
    top_20_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20])

    print("\nTop 20 categories and their product counts:")
    for cat, count in top_20_categories.items():
        print(f"{cat}: {count} products")

    # Create final dataframes for train/test/val
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # Process each category
    for category in tqdm(top_20_categories.keys(), desc="Processing categories"):
        # Get products for this category
        category_df = df[df['categoryName'] == category]

        # Sample up to 10000 products
        sample_size = min(10000, len(category_df))
        sampled_df = category_df.sample(n=sample_size, random_state=42)

        # Split into train/test/val (8:1:1)
        train_size = int(0.8 * sample_size)
        test_size = int(0.1 * sample_size)

        category_train = sampled_df.iloc[:train_size]
        category_test = sampled_df.iloc[train_size:train_size+test_size]
        category_val = sampled_df.iloc[train_size+test_size:]

        train_df = pd.concat([train_df, category_train])
        test_df = pd.concat([test_df, category_test])
        val_df = pd.concat([val_df, category_val])

    # Function to process a single dataset
    def process_split(df, split_name):
        processed_rows = []

        with tqdm(total=len(df), desc=f"Processing {split_name} set") as pbar:
            for idx, row in df.iterrows():
                # Create image filename
                img_filename = f"{split_name}_{idx}.jpg"
                img_path = DATA_DIR / "images" / split_name / img_filename

                # Download image
                if download_image(row['imgUrl'], img_path):
                    # Update image path
                    new_row = row.copy()
                    new_row['imgUrl'] = f"images/{split_name}/{img_filename}"
                    processed_rows.append(new_row)

                pbar.update(1)

        return pd.DataFrame(processed_rows)

    # Process each split
    print("\nDownloading images and processing datasets...")
    final_train_df = process_split(train_df, 'train')
    final_test_df = process_split(test_df, 'test')
    final_val_df = process_split(val_df, 'val')

    # Save processed datasets
    print("\nSaving processed datasets...")
    final_train_df.to_csv(DATA_DIR / 'train.csv', index=False)
    final_test_df.to_csv(DATA_DIR / 'test.csv', index=False)
    final_val_df.to_csv(DATA_DIR / 'val.csv', index=False)

    print("\nProcessing complete!")
    print(f"Final dataset sizes:")
    print(f"Train: {len(final_train_df)} products")
    print(f"Test: {len(final_test_df)} products")
    print(f"Val: {len(final_val_df)} products")

if __name__ == "__main__":
    create_directories()
    process_dataset()
