import pandas as pd
import requests
import os
import hashlib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_session():
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20))
    session.headers.update({"User-Agent": "FYP academic dataset downloader/1.0"})
    return session


HTTP_SESSION = create_session()

def download_image(url, save_dir):
    try:
        filename = hashlib.md5(url.encode()).hexdigest() + '.jpg'
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            return filename

        response = HTTP_SESSION.get(url, timeout=(5, 20))
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return None
            temporary_path = filepath + ".part"
            with open(temporary_path, 'wb') as f:
                f.write(response.content)
            try:
                with Image.open(temporary_path) as image:
                    image.verify()
                os.replace(temporary_path, filepath)
            except (OSError, ValueError):
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
                return None
            return filename
        return None
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

def process_dataset(input_file, image_dir, output_dir):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)

    category_counts = df['categoryName'].value_counts()

    valid_categories = category_counts[(category_counts >= 9000) & (category_counts <= 25000)].index

    print("Downloading images...")
    num_valid_categories = len(valid_categories)
    current_category = 0
    processed_categories = []
    for category in valid_categories:
        category_df = df[df['categoryName'] == category].copy()
        category_df['local_image_path'] = None
        for idx in tqdm(category_df.index, desc=f"Processing category {category} ({current_category+1}/{num_valid_categories})"):
            url = category_df.loc[idx, 'imgUrl']
            filename = download_image(url, image_dir)
            if filename:
                category_df.loc[idx, 'local_image_path'] = filename

        category_df['imgUrl'] = category_df['local_image_path']
        category_df = category_df.drop('local_image_path', axis=1)
        processed_categories.append(category_df.dropna(subset=['imgUrl']))
        current_category += 1

    if not processed_categories:
        raise RuntimeError("No categories matched the configured size limits")
    df_success = pd.concat(processed_categories, ignore_index=True)
    print(f"Total selected rows before filtering: {sum(len(df[df['categoryName'] == c]) for c in valid_categories)}")
    print(f"Successfully downloaded images: {len(df_success)}")
    print(f"Failed downloads: {len(df) - len(df_success)}")

    train_dfs = []
    val_dfs = []
    test_dfs = []

    print("\nCategory distribution before split:")
    print(df_success['categoryName'].value_counts())

    for category in df_success['categoryName'].unique():
        category_df = df_success[df_success['categoryName'] == category]

        train_df, temp_df = train_test_split(category_df, train_size=0.8, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42)

        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

    final_train_df = pd.concat(train_dfs)
    final_val_df = pd.concat(val_dfs)
    final_test_df = pd.concat(test_dfs)

    print("\nFinal dataset statistics:")
    print(f"Train set size: {len(final_train_df)}")
    print(f"Validation set size: {len(final_val_df)}")
    print(f"Test set size: {len(final_test_df)}")

    print("\nCategory distribution in train set:")
    print(final_train_df['categoryName'].value_counts())
    print("\nCategory distribution in validation set:")
    print(final_val_df['categoryName'].value_counts())
    print("\nCategory distribution in test set:")
    print(final_test_df['categoryName'].value_counts())

    final_train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    final_val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    final_test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    INPUT_FILE = os.environ.get(
        "FYP_SOURCE_DATASET", str(PROJECT_ROOT / "dataset" / "amz_uk_processed_data.csv")
    )
    IMAGE_DIR = os.environ.get("FYP_IMAGE_DIR", str(PROJECT_ROOT / "dataset" / "images"))
    OUTPUT_DIR = os.environ.get("FYP_DATA_DIR", str(PROJECT_ROOT / "dataset"))

    process_dataset(INPUT_FILE, IMAGE_DIR, OUTPUT_DIR)
