import pandas as pd
import os
from pathlib import Path

def print_class_distribution(file_path, dataset_name):
    df = pd.read_csv(file_path)

    class_counts = df['categoryName'].value_counts()

    print(f"\n{dataset_name} set class distribution:")
    print(class_counts.to_string())

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("FYP_DATA_DIR", project_root / "dataset")).resolve()
    train_file = data_dir / "train.csv"
    val_file = data_dir / "val.csv"
    test_file = data_dir / "test.csv"

    print_class_distribution(train_file, "Train")
    print_class_distribution(val_file, "Validation")
    print_class_distribution(test_file, "Test")
