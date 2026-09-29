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

Use Python 3.10 or 3.11. Create a virtual environment and install the required extras:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[ml,ui,dev]"
python -m spacy download en_core_web_sm
```

Large generated files are intentionally excluded from Git. Place datasets under
`dataset/` and generated model/index files under `artifacts/`. Copy
`artifacts/manifest.example.json` to `artifacts/manifest.json` and update it when an
artifact set is built.

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

Audit a generated artifact set:

```powershell
fyp-audit --manifest SIM/manifest.json
```

Build the exact text-to-image index, then the image-and-title content index:

```powershell
python Search/searchV.py
python SIM/compu.py
```

Train and evaluate the offline ResNet50+BERT category classifier:

```powershell
python SIM/SIM.py
python SIM/SIM_evaluation.py
```

Start the desktop application after the retrieval artifacts are available:

```powershell
python code/Interface.py
```

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
