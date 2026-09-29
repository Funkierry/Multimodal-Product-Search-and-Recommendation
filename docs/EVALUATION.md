# Evaluation protocol

Historical metrics included with the original project must not be treated as an
independent test result: the old training script merged the original test set into its
training data and reused its validation loader as a test loader.

## Classification

1. Assign each ASIN to exactly one stable train, validation, or test partition.
2. Fit category mappings and all learned preprocessing on train only.
3. Use validation for scheduler decisions, early stopping, and model selection.
4. Load the selected checkpoint and evaluate the test set once.
5. Report accuracy, weighted F1, macro F1, balanced accuracy, per-class recall, and the
   confusion matrix. Macro F1 is the primary early-stopping metric because `Others`
   dominates the current dataset.

## Search

Create a versioned relevance set containing natural-language queries and judged
relevant ASINs. Report Recall@5/10/20, MRR, nDCG@10, p50/p95 latency, peak memory, and
cold-start time. Evaluate filtered queries separately because post-retrieval filtering
can reduce recall.

## Recommendations

Use a temporal or per-user interaction holdout where possible. Report Recall@K,
nDCG@K, catalog coverage, category diversity, and popularity distribution. Compare
the normalized co-purchase model with popularity and same-category baselines.

## Reproducibility

Record the random seed, source data checksum, split checksum, dependency lock, model
name and weights, hyperparameters, and artifact manifest for every reported run.
