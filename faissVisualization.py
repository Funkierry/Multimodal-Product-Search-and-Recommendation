"""Create a sampled two-dimensional PCA plot of a Faiss index."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=project_root / "Search" / "faiss_index.index",
    )
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "figures" / "faiss_pca.png",
    )
    args = parser.parse_args()

    index = faiss.read_index(str(args.index))
    sample_size = min(max(args.sample_size, 2), index.ntotal)
    sample_indices = np.linspace(0, index.ntotal - 1, sample_size, dtype=np.int64)
    vectors = np.vstack([index.reconstruct(int(index_id)) for index_id in sample_indices])
    vectors_reduced = PCA(n_components=2, random_state=42).fit_transform(vectors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], s=8, alpha=0.55)
    plt.title(f"PCA sample of Faiss vectors (n={sample_size:,})")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
