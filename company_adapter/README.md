# DataShield AI — Layer 2: Company Adapter

Adds **company-specific sensitive data detection** on top of the base LLM classifier.

Admins upload their internal documents → the pipeline extracts sensitive terms, builds a multilingual semantic index, and exports a `company_profile.json` that the browser extension loads automatically.

---

## Folder Structure

```
company_adapter/
├── colab_runner.py           ← Run this to execute the full pipeline
├── requirements.txt          ← Python dependencies
├── adapter_utils.py          ← Shared text extraction utilities
├── 1_extract_terms.py        ← NER + regex + frequency extraction
├── 2_build_embeddings.py     ← Multilingual embeddings + FAISS + PCA
├── 3_profile_builder.py      ← Assembles company_profile.json
├── 4_test_adapter.py         ← Tests profile against sample prompts
├── 5_company_adapter.js      ← Browser JS layer (drop-in for extension)
├── sample_docs/
│   └── sample_company_doc.txt ← Test document (replace with real docs)
└── README.md
```

**Output (after running):**
```
adapter_output/
├── extracted_terms.json       ← Raw terms from documents
├── enriched_terms.json        ← After embedding + dedup + scoring
├── company_profile.json       ← DEPLOY THIS to the extension
└── company_profile_lite.json  ← Human-readable version (no vectors)
```

---

## Quick Start (Google Colab)

```python
# 1. Clone repo and enter directory
!git clone https://github.com/chaima-m/Data-Shield-.git
%cd Data-Shield-

# 2. Upload your company documents to Colab
# (or use the included sample_docs/ for testing)
from google.colab import files
uploaded = files.upload()  # Upload PDF, DOCX, TXT, XLSX, CSV files

# Move uploaded files to docs folder
!mkdir -p my_company_docs
import shutil
for fname in uploaded:
    shutil.move(fname, f"my_company_docs/{fname}")

# 3. Run the full pipeline
%run company_adapter/colab_runner.py \
    --docs_dir my_company_docs \
    --company_id acme_corp \
    --threshold 0.82

# 4. Download the profile
from google.colab import files
files.download('adapter_output/company_profile.json')
```

---

## Step-by-Step (Manual)

```bash
# Install dependencies
pip install -r company_adapter/requirements.txt
python -m spacy download xx_ent_wiki_sm

# Step 1: Extract terms from documents
python company_adapter/1_extract_terms.py --docs_dir ./my_company_docs

# Step 2: Build embeddings (downloads ~90MB model first time)
python company_adapter/2_build_embeddings.py

# Step 3: Build profile
python company_adapter/3_profile_builder.py --company_id acme_corp --threshold 0.82

# Step 4: Test
python company_adapter/4_test_adapter.py
```

---

## Detection Layers

| Layer | Where | How |
|---|---|---|
| **2a — Exact match** | JS + Python | Case-insensitive substring match against extracted entity names |
| **2b — Pattern match** | JS + Python | Compiled regex for internal codes (e.g. `HELIOS[-_]?Q4`) |
| **2c — Keyword density** | JS + Python | ≥2 sensitive keywords in a single prompt → flagged |
| **2d — Semantic similarity** | JS (n-gram) | Character n-gram cosine similarity vs stored term vectors |

All four run in parallel. **Any layer flagging → prompt blocked.**

---

## What the Profile JSON Contains

```json
{
  "company_id": "acme_corp",
  "detection_rules": {
    "exact_match":   [{"text": "Project Helios", "type": "WORK_OF_ART"}],
    "pattern_match": [{"pattern": "HELIOS[-_]?Q4", "label": "confidential"}],
    "keyword_match": ["helios", "novatel", "operation sunrise", ...]
  },
  "semantic_index": {
    "terms":   ["Project Helios", "NovaTech deal", ...],
    "labels":  ["confidential", "confidential", ...],
    "vectors": [[...64-dim floats...]],
    "pca_components": [...],
    "pca_mean": [...]
  }
}
```

---

## Linking to the Extension

Replace in `4_detection_engine.js`:

```javascript
// BEFORE
import { CompanyRuleEngine } from './some_file.js';

// AFTER  
import { EnhancedCompanyEngine as CompanyRuleEngine } from './5_company_adapter.js';
```

The `EnhancedCompanyEngine` is a drop-in replacement — same `loadProfile()` and `analyze()` API, but with the added semantic layer.

---

## Tuning

| Problem | Fix |
|---|---|
| Missing company-specific terms | Add more documents in Step 1 |
| Too many false positives | Raise threshold to 0.88 or remove noisy terms from profile_lite.json |
| Arabic terms not detected | Ensure Arabic documents are included; spaCy xx model has basic Arabic support |
| Profile file too large (>500KB) | Reduce `MAX_SEMANTIC_VECTORS` in `3_profile_builder.py` |
