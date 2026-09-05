# Phase 4: NER

The rule-based extractor is the verified baseline and supplies weak-label candidates. BIO tagging will represent tokens as `B-SKILL`, `I-SKILL`, or `O`. A transformer fine-tuning run is not claimed yet: it requires a labeled train/validation/test corpus and will report measured held-out precision and recall only after execution.

The stable schema already allows a future NER adapter to replace or supplement rules without changing downstream consumers.
