"""Audit fixed classifier partitions before loading either pretrained model."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {"asin", "title", "imgUrl", "categoryName"}


def read_split(path):
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin1")
    frame.columns = [name.lstrip("\ufeff").removeprefix("ï»¿") for name in frame.columns]
    return frame


def audit_frames(frames):
    """Return counts and reject product leakage or malformed training inputs."""
    report = {"splits": {}, "overlap": {}}
    ids = {}
    for name in SPLITS:
        frame = frames[name]
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} is empty")
        if frame[["asin", "title", "imgUrl"]].isna().any().any():
            raise ValueError(f"{name} contains null product identifiers, titles, or images")
        if frame["asin"].duplicated().any():
            raise ValueError(f"{name} contains duplicate ASINs")
        ids[name] = set(frame["asin"])
        report["splits"][name] = {
            "rows": len(frame),
            "unique_classes": int(frame["categoryName"].nunique()),
            "missing_category_rows": int(frame["categoryName"].isna().sum()),
        }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[left] & ids[right]
        report["overlap"][f"{left}_{right}"] = len(overlap)
        if overlap:
            raise ValueError(f"{left} and {right} share {len(overlap)} ASINs")

    trained_classes = set(frames["train"]["categoryName"].dropna())
    report["train_singleton_classes"] = int(
        (frames["train"]["categoryName"].value_counts() == 1).sum()
    )
    for name in ("val", "test"):
        unseen = ~frames[name]["categoryName"].isin(trained_classes)
        report["splits"][name]["unseen_class_rows"] = int(unseen.sum())
        report["splits"][name]["unseen_classes"] = int(
            frames[name].loc[unseen, "categoryName"].nunique()
        )
        report["splits"][name]["known_class_coverage"] = round(
            1 - float(unseen.mean()), 6
        )
    return report


def audit_directory(data_dir):
    paths = {name: data_dir / f"{name}_updated.csv" for name in SPLITS}
    frames = {name: read_split(path) for name, path in paths.items()}
    report = audit_frames(frames)
    report["files"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    }
    alternate_path = data_dir / "train_updated_final.csv"
    if alternate_path.is_file():
        alternate = read_split(alternate_path)
        if {"asin", "categoryName"}.issubset(alternate.columns):
            comparison = frames["train"][["asin", "categoryName"]].merge(
                alternate[["asin", "categoryName"]],
                on="asin", how="inner", suffixes=("_raw", "_final"), validate="one_to_one",
            )
            report["alternative_train_labels"] = {
                "path": str(alternate_path),
                "sha256": hashlib.sha256(alternate_path.read_bytes()).hexdigest(),
                "rows": len(alternate),
                "matched_asins": len(comparison),
                "unique_classes": int(alternate["categoryName"].nunique()),
                "changed_labels": int(
                    (comparison["categoryName_raw"] != comparison["categoryName_final"]).sum()
                ),
            }
    return frames, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, report = audit_directory(args.data_dir)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
