# Generated artifacts

This directory is reserved for locally generated model checkpoints, embeddings, Faiss
indexes, metadata, and their manifest. Large binary files are ignored by Git.

Every artifact build should produce `manifest.json` with dataset/model identity and
row counts. The application must reject artifacts whose counts or vector dimensions do
not match instead of silently truncating them.
