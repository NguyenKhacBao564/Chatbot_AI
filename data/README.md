## Overview
JSON file `faq_cleaned.json` (~100-200 Q&A) used by the retrieval pipeline.

## How to preprocess
```bash
python scripts/preprocess_faq.py --input data/raw/faq_raw.json --output data/processed/faq_cleaned.json