# Multimodal Product Search and Recommendation

This repository contains the reproducible source code for the project described in
`FYP reports DC127910 CUI JIANFU -Final Version.pdf`. The application combines
OpenCLIP retrieval, item-to-item similarity, review summaries, and interaction-based
recommendations.

## Security notice

Never commit browser cookies, access tokens, datasets, downloaded product images,
model checkpoints, or generated vector indexes. The repository ignores these files by
default. Use the environment variables in `.env.example` for local paths.

## Setup

Use Python 3.10 through 3.12. Create a virtual environment and install the required extras:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[ml,ui,dev]"
```

Large generated files are intentionally excluded from Git. Place datasets under
`dataset/`. The legacy builders write the search index and ordered catalog to
`Search/`, then the content index and validated manifest to `SIM/`. The two builders
must run in that order. `FYP_DATA_DIR` can relocate the source dataset; keep its value
the same when building indexes and running the desktop app.

## Expected data

The training pipeline expects CSV files with at least these columns:

- `asin`, `title`, `imgUrl`, `categoryName`
- optional catalog fields: `price`, `stars`, `reviews`, `productURL`
- interaction data: `ASIN`, `user`, and optionally `Production`

Paths may be absolute or relative to `FYP_DATA_DIR`. Generated metadata should store
paths relative to the project or data directory so the project remains portable.

## Validation

Run the lightweight tests without downloading ML models:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

GitHub CI runs these tests and checks all installed command-line entry points on
Python 3.10, 3.11, and 3.12.

Audit a generated artifact set:

```powershell
fyp-audit --manifest SIM/manifest.json
```

Evaluate the interaction-based recommender against popularity and same-category
baselines using the project's simulated interaction file:

```powershell
$env:FYP_DATA_DIR = "D:\path\to\original\dataset"
fyp-evaluate-recommendations
```

See `docs/EVALUATION.md` for the reported baseline, the search judgment format, and
the limits of these offline measurements.

Build the exact text-to-image index, then the image-and-title content index:

```powershell
python Search/searchV.py --rebuild --build-only
python SIM/compu.py
```

The builders require unique ASINs and readable images. Catalog image paths are saved
relative to `FYP_DATA_DIR`; missing images during the content build stop the build
instead of producing zero vectors. Artifacts produced before manifest schema 2 must
be rebuilt or migrated before starting the desktop app. To migrate existing exact
Faiss indexes without rerunning OpenCLIP, use the original dataset directory:

```powershell
fyp-migrate-artifacts --data-dir "D:\path\to\original\dataset"
```

Migration checks all image paths, index rows, and content vectors. It keeps the old
metadata as `Search/items_meta.legacy.json` and writes the new search embedding
matrix and `SIM/manifest.json`. The source dataset can stay outside this repository;
set `FYP_DATA_DIR` to the same directory when launching the app.

Train and evaluate the offline ResNet50+BERT category classifier:

```powershell
$env:FYP_DATA_DIR = "D:\path\to\dataset"
$env:FYP_BATCH_SIZE = "32" # measured at about 3 GB for one training batch on an RTX 4060
$env:FYP_EPOCHS = "30"
$env:HF_HOME = (Join-Path (Get-Location) "results\model_cache\hf")
$env:TORCH_HOME = (Join-Path (Get-Location) "results\model_cache\torch")
python SIM/classification_audit.py --data-dir $env:FYP_DATA_DIR
python SIM/SIM.py
python SIM/SIM_evaluation.py
```

Activate the environment containing CUDA-enabled PyTorch first; the `ml` extra also
requires `transformers`. The classifier uses validation for checkpoint selection and
evaluates the fixed test partition after training. Test categories absent from train
count as errors. This command trains on the raw fine-grained categories; it does not
replicate the historical 11-category experiment. See
[the evaluation protocol](docs/EVALUATION.md) before comparing results with the PDF.

Start the desktop application after the retrieval artifacts are available:

```powershell
python code/Interface.py
```

Review aspect summaries are built offline. If `reviews_data.json` is absent, the UI
still opens and shows a no-data message in the sentiment panel. To generate it:

```powershell
python -m pip install -e ".[reviews]"
python -m spacy download en_core_web_sm
fyp-build-reviews --input "D:\path\to\dataset\Dataset_Rec.csv" --output reviews_data.json
```

The UI only reads `reviews_data.json`; it no longer loads spaCy or processes the
review CSV during startup. The builder writes the JSON atomically after validating
the source columns.

Retrieval timing and memory measurements for the local catalog are in
[the performance report](docs/PERFORMANCE.md). The UI checks artifact checksums at
startup; run `fyp-audit --manifest SIM/manifest.json` for the full row-by-row index
check after rebuilding artifacts.

The legacy desktop entry point remains `code/Interface.py`. It requires the `ml` and
`ui` extras and generated indexes. Training and final evaluation must use disjoint
train, validation, and test sets; the test set is evaluated only after model selection.

## Repository layout

```text
src/fyp/                 reusable configuration and domain logic
Preprocess/              dataset preparation scripts
Search/                  retrieval index builder and legacy search UI
SIM/                     multimodal training and evaluation
REC/                     recommendation experiments
code/                    desktop application
tests/                   fast deterministic tests
artifacts/               local generated artifacts (ignored by Git)
```
