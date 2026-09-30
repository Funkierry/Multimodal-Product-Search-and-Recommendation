# Local retrieval performance

Measurements below used the 198,910-item, 768-dimensional exact Faiss index on the
project laptop. They are engineering measurements, not search-quality scores.

## Single-query Faiss search

Twenty saved image vectors were used as queries with `k=200`. They isolate index
search from OpenCLIP text encoding and UI rendering. On this machine, loading the
search index took 0.59 seconds and increased process RSS by about 583 MB.

| CPU threads | p50 (ms) | p95 (ms) |
| ---: | ---: | ---: |
| 1 | 31.87 | 33.25 |
| 2 | 18.97 | 23.62 |
| 4 | 16.13 | 16.97 |
| 8 | 13.12 | 28.13 |
| 16 | 29.16 | 35.37 |

The UI and live evaluator now default to four Faiss threads. Set
`FYP_FAISS_THREADS` to tune for another CPU. No approximate index is warranted by
these measurements alone; it would need a judged Recall@K comparison.

## Artifact and runtime loading

Full `ArtifactManifest.load()` validation took 15.82 seconds and reached 2,580 MB
peak RSS in a fresh process. Omitting the row-by-row Faiss reconstruction check,
while retaining file checksums, metadata validation, matrix shape, and vector norm
checks, took 8.14 seconds and reached 1,397 MB peak RSS. The UI now uses this
runtime validation mode; `fyp-audit` retains the complete row check.

In a smoke run with OpenCLIP model initialization replaced by a no-op, the optimized
retrieval service loaded the actual catalog and search index in 22.88 seconds,
increasing RSS by 848 MB. The content index remained unloaded until the first
similar-item request. That request took 1.26 seconds and raised the total RSS
increase to 1,432 MB. These numbers exclude model initialization and do not
represent full desktop startup time. The loader resolves the three image
directories once instead of resolving each of the 198,910 file paths individually.
