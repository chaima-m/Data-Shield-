# DataShield AI — LLM Component: Implementation Guide

## File Structure

```
datashield_llm/
├── 1_merge_datasets.py           # Merge all datasets into unified format
├── 2_finetune.py                  # Fine-tune + ONNX export pipeline
├── 3_browser_classifier.js        # In-browser ONNX inference (extension)
├── 4_detection_engine.js          # Combined 3-layer detection (extension)
├── 5_admin_profile_generator.py   # Admin panel for company profile upload
└── README.md
```

---

## Step 1 — Build the Dataset

```bash
pip install datasets pandas scikit-learn tqdm transformers
python 1_merge_datasets.py
# Output: ./data/datashield_dataset.csv  (~15-20k rows)
```

The script:
- Downloads **AI4Privacy** and **MultiNERD** from HuggingFace
- Tries **CANERCorpus** (Arabic NER)
- Generates synthetic samples from realistic templates for all 6 labels × 3 languages
- Deduplicates and upsamples minority cells to ≥600 samples each

**Dataset format:**
```json
{"text": "My SSN is 123-45-6789", "label": "pii", "language": "en"}
{"text": "IBAN FR7630006000011234567890189", "label": "financial", "language": "fr"}
{"text": "كلمة المرور هي Admin@2024", "label": "credentials", "language": "ar"}
```

**Labels:** `safe` | `pii` | `financial` | `confidential` | `health` | `credentials`

---

## Step 2 — Fine-tune

```bash
pip install transformers datasets torch scikit-learn evaluate \
            onnx onnxruntime optimum[onnxruntime] accelerate

# Default (DistilmBERT — recommended, ~250MB, fast CPU)
python 2_finetune.py

# If Arabic F1 < 0.80, upgrade:
python 2_finetune.py --model xlm-roberta-base

# Quick test run:
python 2_finetune.py --epochs 2 --batch 8
```

**Outputs:**
- `./model_output/final/`         — PyTorch fine-tuned model
- `./model_onnx/`                 — ONNX fp32 export
- `./model_onnx_quantized/`       — **INT8 quantized ONNX** ← use this in browser

**Target metrics (validation set):**
| Metric | Target |
|--------|--------|
| Overall weighted F1 | > 0.85 |
| Arabic F1 | > 0.80 |
| False positive rate (safe) | < 10% |

---

## Step 3 — Integrate into Browser Extension

Copy the quantized model files to your extension's `/models/` directory:
```
model_onnx_quantized/
├── model_quantized.onnx   ← main model file
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── special_tokens_map.json
```

Install JS dependencies:
```bash
npm install onnxruntime-web franc-min
```

In your **background service worker** (`background.js`):
```javascript
import { detectionEngine } from './4_detection_engine.js';

// Initialize once on extension startup
await detectionEngine.initialize();

// In message handler (from content script):
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'ANALYZE_PROMPT') {
    detectionEngine.analyze(msg.text).then(sendResponse);
    return true; // keep channel open for async response
  }
});
```

In your **content script** (when user types in ChatGPT/Claude/Gemini):
```javascript
// Debounced analysis on input
inputEl.addEventListener('input', async () => {
  const result = await chrome.runtime.sendMessage({
    type: 'ANALYZE_PROMPT',
    text: inputEl.value
  });
  
  if (result.isSensitive) {
    showAlert(result);      // your UI overlay
    pauseHeartbeat();       // stops proxy from allowing traffic
  }
});
```

### Detection layers:
| Layer | Source | When it fires |
|-------|--------|---------------|
| **Pattern** | Regex + Luhn | Credit cards, SSN, API keys, DB strings |
| **LLM** | ONNX DistilmBERT | Confidence ≥ 0.65 |
| **Company** | Admin profile | Exact match, code patterns, keyword density ≥ 2 |

---

## Step 4 — Admin Profile Generator (Enterprise)

```bash
pip install flask pymupdf python-docx pandas spacy cryptography \
            sentence-transformers faiss-cpu werkzeug
python -m spacy download xx_ent_wiki_sm

python 5_admin_profile_generator.py
# Open: http://127.0.0.1:8889
```

The admin:
1. Uploads company documents (PDF, DOCX, TXT, XLSX, CSV)
2. Reviews extracted terms, removes false positives, adds custom terms
3. Adjusts sensitivity threshold (0.82 recommended)
4. Clicks **Apply & Export**  →  `company_profiles/my_company_profile.json`
5. IT deploys `profile.json` to employee machines via Group Policy

The extension's `CompanyRuleEngine` loads this profile from `chrome.storage.local` on startup.

**Deployment via Group Policy:**
```powershell
# Run on each employee machine (PowerShell as admin)
$profileJson = Get-Content "\\fileserver\datashield\my_company_profile.json" -Raw
$regPath = "HKLM:\SOFTWARE\DataShield"
New-Item -Path $regPath -Force | Out-Null
Set-ItemProperty -Path $regPath -Name "CompanyProfile" -Value $profileJson
```

---

## Recommended Sprint Allocation

| Week | Task |
|------|------|
| 12 | Run `1_merge_datasets.py`, review distribution, add more Arabic synthetic data |
| 13 | Run `2_finetune.py`, check per-language F1, iterate on dataset if needed |
| 14 | Copy ONNX model to extension, wire up `3_browser_classifier.js` |
| 15 | Integrate `4_detection_engine.js`, test with `5_admin_profile_generator.py` |

---

## Troubleshooting

**Arabic F1 too low (< 0.80):**
- Add more Arabic samples — run `1_merge_datasets.py --synth-mult 150`
- Switch to XLM-RoBERTa: `python 2_finetune.py --model xlm-roberta-base`

**Model too slow in browser (> 150ms):**
- Ensure you're using the quantized model (`model_quantized.onnx`)
- Reduce `MAX_LENGTH` from 256 to 128 in `3_browser_classifier.js`
- Enable WASM SIMD in `ort.env.wasm.simd = true`

**High false positive rate on "safe":**
- Lower the LLM confidence threshold from 0.65 to 0.70 in `4_detection_engine.js`
- Add more "safe" training examples, especially in the domain your users work in

**Model file > 100MB after quantization:**
- This is expected for XLM-RoBERTa (125M params). For DistilmBERT it should be ~65-80MB.
- Use `DistilBERT` if size is a hard constraint.
