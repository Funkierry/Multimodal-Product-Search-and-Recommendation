# Evaluation protocol

Historical metrics included with the original project must not be treated as an
independent test result: the old training script merged the original test set into its
training data and reused its validation loader as a test loader.

## Classification

1. Assign each ASIN to exactly one stable train, validation, or test partition.
2. Fit category mappings and all learned preprocessing on train only.
3. Use validation for scheduler decisions, early stopping, and model selection.
4. Load the selected checkpoint and evaluate the test set once.
5. Report accuracy, weighted F1, macro F1, balanced accuracy, per-class recall, and a
   sparse confusion matrix. Macro F1 is the primary early-stopping metric because the
   current category distribution is highly imbalanced.

The local fixed split contains 159,124 train, 19,900 validation, and 19,886 test
rows, with no ASIN overlap. The train set has 6,523 classes, including 4,168 with a
single sample. One train row has no category and is explicitly excluded from
training. Of the validation and test rows, 521 and 531 respectively have categories
absent from train. These are never silently discarded from the final test metric:
they receive the sentinel truth label `-1` and count as incorrect. Validation uses
only known classes for model selection, and its coverage is recorded separately.

There are **two different label schemes** in the local data. `train_updated.csv`
uses the 6,523 raw categories, while `train_updated_final.csv` and the current search
catalog use 11 broad categories. The two train files share all 159,124 ASINs but
159,119 labels differ. The repaired training script explicitly runs the raw-category
experiment and warns about this distinction. Its future metrics cannot be compared
with the historical 11-class result. No independently annotated 11-class validation
or test CSV is currently present; catalog labels for those partitions are derived
labels, not independently verified ground truth.

Run `python SIM/classification_audit.py --data-dir path/to/dataset` before training.
Training writes `results/classification_data_audit.json` with split counts, the
alternate train label scheme, and source CSV checksums. Final testing writes
`results/classification_test_report.json` with coverage, metrics, per-class recall,
and nonzero confusion entries. The 2025 model
checkpoint lacks the new split-protocol metadata; the standalone evaluator refuses
to report it as an independent test result. No independent classifier accuracy has
been reported yet from the repaired training pipeline.

## Search

Create a versioned relevance set containing natural-language queries and judged
relevant ASINs. Report Recall@5/10/20, MRR, nDCG@10, p50/p95 latency, peak memory, and
cold-start time. Evaluate filtered queries separately because post-retrieval filtering
can reduce recall.

The evaluator accepts one manually judged query per JSONL line. This is an illustrative
format, not a scored dataset:

```jsonl
{"query_id":"q001","query":"red waterproof jacket","relevant_asins":["REVIEWED_ASIN_1"]}
{"query_id":"q002","query":"black ankle boots","relevant_asins":["REVIEWED_ASIN_2"],"filters":{"category":"Boots","max_price":100}}
```

Judge relevance against the product catalog before looking at this system's ranking.
Use the same judgments for every search baseline. Once reviewed queries exist, run:

```powershell
fyp-evaluate-search --judgments path/to/judgments.jsonl --output results/search_eval.json --save-rankings results/search_rankings.jsonl
```

The evaluator checks the schema 2 artifact manifest, runs OpenCLIP text queries over
the exact Faiss index, applies any category or price filters to the candidate pool,
and reports Recall@5/10/20, MRR, nDCG, query p50/p95 latency, and cold-start time.
It groups filtered and unfiltered queries separately. Rankings saved from a prior run
can be rescored with `--rankings results/search_rankings.jsonl` without loading the
model. Search quality is not reported yet because no independently judged query set
has been created. Peak memory is also not currently measured by this evaluator.

## Recommendations

Use a temporal or per-user interaction holdout where possible. Report Recall@K,
nDCG@K, catalog coverage, category diversity, and popularity distribution. Compare
the normalized co-purchase model with popularity and same-category baselines.

The current `Dataset_Rec.csv` has no timestamps and contains simulated user-item
interactions according to the project report. We therefore use a deterministic
per-user leave-one-out split, keeping the held-out item in the training catalog.
Users with fewer than two items or no eligible warm item are excluded. The model
aggregates item-to-item co-purchase scores from the user's remaining items and uses
training popularity to fill empty recommendation slots. The same-category baseline
uses category labels from `Search/items_meta.json`; the interaction file's
`Production` field is not treated as an item category.
Category metadata is available for 25 of the 26 interacted products.

The first run used 476 unique interactions, 178 users, and 26 items. Of these, 133
users were evaluated and 45 were excluded. Results at seed 42:

| Method | Recall@5 | Recall@10 | nDCG@10 | Catalog coverage@10 |
| --- | ---: | ---: | ---: | ---: |
| Normalized co-purchase with popularity fallback | 0.226 | 0.481 | 0.206 | 1.000 |
| Popularity | 0.203 | 0.459 | 0.223 | 0.577 |
| Same category with popularity fallback | 0.263 | 0.481 | 0.230 | 0.692 |

Co-purchase has slightly higher Recall@10 than popularity but lower nDCG@10 than
both baselines. These are small, simulated, warm-item results and do not establish
performance on real user behavior or cold-start products. The complete local report
is generated with `fyp-evaluate-recommendations` and includes dataset, catalog, and
split checksums. The split checksum for this run is
`ae23ef4fc00ba2b20e0e861fbc4fac22f88b06eaa2acc2843a612fd3c666536d`.
The interaction file SHA-256 is
`33b5cc353c4555cf2c3c7b64396da710876f1261997daf6f8448a4db32412c11`;
the catalog metadata SHA-256 is
`ee2849f636d59ebf7150392b98ad7089e87199e635d3d8dcfd85a459a0384d9f`.

## Reproducibility

Record the random seed, source data checksum, split checksum, dependency lock, model
name and weights, hyperparameters, and artifact manifest for every reported run.
