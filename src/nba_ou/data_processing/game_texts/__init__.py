"""Game texts (NBA.com written reports, recap transcripts) and their embeddings.

``team_perspective`` and ``players`` rewrite a text from either team's point of
view; ``documents`` reads the archived texts; ``embedding_inputs`` turns a report
into the exact model inputs; ``nebius`` embeds them with Qwen3-Embedding-8B;
``vector_codec`` and ``embedding_store`` keep the vectors in S3 without loss;
``embed_reports`` ties them together. See ``docs/pipeline/text_embeddings.md``.
"""
