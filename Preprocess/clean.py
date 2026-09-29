import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", PROJECT_ROOT / "dataset")).resolve()

def preprocess_csv(input_file, output_file, images_dir):
    df = pd.read_csv(input_file)
    initial_length = len(df)

    def process_imgurl(imgurl):
        if pd.isnull(imgurl):
            return None
        if ';' in imgurl:
            return imgurl.split(';')[0].strip()
        return imgurl.strip()

    df['imgUrl'] = df['imgUrl'].apply(process_imgurl)

    def verify_image(imgurl):
        if pd.isnull(imgurl):
            return False
        return os.path.exists(os.path.join(images_dir, os.path.basename(imgurl)))

    df['valid_image'] = df['imgUrl'].apply(verify_image)

    cleaned_df = df[df['valid_image']].drop(columns=['valid_image'])
    cleaned_length = len(cleaned_df)
    print(f"Processed {input_file}: {initial_length} -> {cleaned_length} samples after cleaning.")


    cleaned_df.to_csv(output_file, index=False)
    print(f"Cleaned CSV saved to {output_file}")

if __name__ == '__main__':
    preprocess_csv(
        input_file=DATA_DIR / "train.csv",
        output_file=DATA_DIR / "train_clean.csv",
        images_dir=DATA_DIR / "images" / "train"
    )
    preprocess_csv(
        input_file=DATA_DIR / "val.csv",
        output_file=DATA_DIR / "val_clean.csv",
        images_dir=DATA_DIR / "images" / "val"
    )
    preprocess_csv(
        input_file=DATA_DIR / "test.csv",
        output_file=DATA_DIR / "test_clean.csv",
        images_dir=DATA_DIR / "images" / "test"
    )
