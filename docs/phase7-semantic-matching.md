# Phase 7: Semantic Matching

The verified fallback uses normalized token vectors and cosine similarity. It makes the mechanism inspectable without claiming sentence-transformer quality. The production adapter will use Sentence Transformers after its model download and environment are verified; no similarity quality metric is reported yet.
