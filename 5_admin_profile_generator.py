"""
DataShield AI — Admin Profile Generator
========================================
Run by the IT admin (locally) to extract sensitive terms from company documents
and generate an encrypted detection profile for employee deployment.

Install:
    pip install flask pymupdf python-docx pandas spacy cryptography \
                sentence-transformers faiss-cpu werkzeug
    python -m spacy download xx_ent_wiki_sm

Run:
    python 5_admin_profile_generator.py
    Open: http://127.0.0.1:8889
"""

import os
import re
import json
import hashlib
import base64
import logging
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional

import fitz                          # PyMuPDF
from docx import Document as DocxDoc
import pandas as pd
import spacy
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────

UPLOAD_DIR  = Path("./admin_uploads")
PROFILE_DIR = Path("./company_profiles")
STATIC_DIR  = Path("./admin_static")

for d in (UPLOAD_DIR, PROFILE_DIR, STATIC_DIR):
    d.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "xlsx", "csv"}

# Common English/French/Arabic stop words to ignore in keyword extraction
STOP_WORDS = {
    "the","and","for","that","this","with","from","have","will","been","they",
    "their","company","which","would","about","there","when","your","email",
    "also","more","than","into","other","some","time","very","just","over",
    "such","only","most","must","each","both","here","after","even","many",
    "these","those","then","well","were","what","like","said","make","name",
    "les","des","une","dans","pour","avec","est","que","qui","pas","sur",
    "tout","mais","par","plus","nous","vous","ils","elle","être","avoir",
    "من","في","على","هو","هي","إلى","عن","مع","قد","لا","ما","كان"
}

# ─── NLP Setup ─────────────────────────────────────────────────────────────────

try:
    nlp = spacy.load("xx_ent_wiki_sm")
    NLP_AVAILABLE = True
except OSError:
    log.warning("spaCy multilingual model not found. Run: python -m spacy download xx_ent_wiki_sm")
    NLP_AVAILABLE = False

# ─── File Extraction ──────────────────────────────────────────────────────────

def extract_text(filepath: Path) -> str:
    ext = filepath.suffix.lower().lstrip(".")
    try:
        if ext == "pdf":
            doc = fitz.open(str(filepath))
            return "\n".join(page.get_text("text") for page in doc)

        elif ext == "docx":
            doc = DocxDoc(str(filepath))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext == "txt":
            return filepath.read_text(encoding="utf-8", errors="ignore")

        elif ext == "xlsx":
            df = pd.read_excel(filepath, dtype=str)
            return df.fillna("").to_string(index=False)

        elif ext == "csv":
            df = pd.read_csv(filepath, dtype=str, on_bad_lines="skip")
            return df.fillna("").to_string(index=False)

    except Exception as e:
        log.error(f"Failed to extract {filepath.name}: {e}")
    return ""


# ─── Term Extraction ──────────────────────────────────────────────────────────

def extract_named_entities(text: str) -> list[dict]:
    """Extract ORG / PRODUCT / WORK_OF_ART / LAW named entities via spaCy."""
    if not NLP_AVAILABLE:
        return []
    keep_types = {"ORG", "PRODUCT", "WORK_OF_ART", "LAW", "GPE", "FAC"}
    doc = nlp(text[:500_000])  # spaCy limit guard
    seen = set()
    entities = []
    for ent in doc.ents:
        name = ent.text.strip()
        if ent.label_ in keep_types and len(name) > 3 and name not in seen:
            seen.add(name)
            entities.append({"text": name, "type": ent.label_})
    return entities


def extract_code_patterns(text: str) -> list[str]:
    """
    Extract internal code patterns that look like:
      PROJECT_NAME, ClientXK2, Operation-Alpha, DOC-2024-001
    """
    patterns = re.findall(
        r'\b[A-Z][A-Z0-9]{1,}[-_][A-Z0-9]{2,}\b'      # CODE_123
        r'|\b[A-Z]{2,}[-_][A-Za-z0-9]{3,}\b'           # OP-Sunrise
        r'|\b(?:Project|Operation|Initiative|Phase)\s+[A-Z][A-Za-z]+\b',  # Project Helios
        text
    )
    return list(set(p.strip() for p in patterns if len(p) > 4))


def extract_keywords(text: str, top_n: int = 150, min_freq: int = 3) -> list[str]:
    """
    Extract high-frequency domain-specific nouns/terms via simple TF counting.
    Filters out stop words and common English words.
    """
    words = re.findall(r'\b[A-Za-z\u0600-\u06FF]{5,}\b', text)
    freq  = Counter(w.lower() for w in words if w.lower() not in STOP_WORDS)
    return [word for word, count in freq.most_common(top_n) if count >= min_freq]


def run_extraction(text: str) -> dict:
    return {
        "named_entities": extract_named_entities(text),
        "custom_patterns": extract_code_patterns(text),
        "sensitive_keywords": extract_keywords(text),
    }


# ─── Encryption ───────────────────────────────────────────────────────────────

def _derive_fernet_key(device_secret: str) -> bytes:
    """
    Derive a Fernet key from an admin-provided device secret / company passphrase.
    In production this should match the device-bound key from the proxy architecture.
    """
    digest = hashlib.sha256(device_secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_profile(profile: dict, passphrase: str) -> bytes:
    key   = _derive_fernet_key(passphrase)
    f     = Fernet(key)
    data  = json.dumps(profile, ensure_ascii=False).encode("utf-8")
    return f.encrypt(data)


def decrypt_profile(encrypted: bytes, passphrase: str) -> dict:
    key  = _derive_fernet_key(passphrase)
    f    = Fernet(key)
    data = f.decrypt(encrypted)
    return json.loads(data.decode("utf-8"))


# ─── Profile Builder ──────────────────────────────────────────────────────────

def build_profile(
    all_terms:   dict,
    company_id:  str,
    threshold:   float = 0.82,
    passphrase:  Optional[str] = None,
    extra_terms: Optional[list] = None,
) -> dict:
    """Build, optionally encrypt, and save the company detection profile."""

    if extra_terms:
        for t in extra_terms:
            all_terms["named_entities"].append({"text": t.strip(), "type": "CUSTOM"})

    # Deduplicate
    seen_entities = set()
    unique_entities = []
    for e in all_terms["named_entities"]:
        key = e["text"].lower()
        if key not in seen_entities:
            seen_entities.add(key)
            unique_entities.append(e)

    profile = {
        "company_id":   company_id,
        "version":      "1.0",
        "created_at":   datetime.utcnow().isoformat() + "Z",
        "threshold":    threshold,
        "detection_rules": {
            "exact_match":    unique_entities,
            "pattern_match":  [
                {"pattern": re.escape(p), "label": "confidential"}
                for p in set(all_terms["custom_patterns"])
            ],
            "keyword_match":  list(set(all_terms["sensitive_keywords"])),
        },
        "stats": {
            "exact_terms":    len(unique_entities),
            "regex_patterns": len(all_terms["custom_patterns"]),
            "keywords":       len(all_terms["sensitive_keywords"]),
        },
    }

    # Save unencrypted JSON (for review)
    profile_path = PROFILE_DIR / f"{company_id}_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    # Save encrypted version if passphrase provided
    if passphrase:
        enc_path = PROFILE_DIR / f"{company_id}_profile.enc"
        enc_data = encrypt_profile(profile, passphrase)
        with open(enc_path, "wb") as f:
            f.write(enc_data)
        log.info(f"Encrypted profile → {enc_path}")

    log.info(f"Profile saved → {profile_path}")
    return profile


# ─── Flask Admin UI ───────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(STATIC_DIR))

@app.route("/")
def index():
    return ADMIN_HTML  # served inline below


@app.route("/upload", methods=["POST"])
def upload_files():
    files      = request.files.getlist("documents")
    company_id = request.form.get("company_id", "company")
    passphrase = request.form.get("passphrase", "")

    all_terms = {"named_entities": [], "custom_patterns": [], "sensitive_keywords": []}
    processed = []
    errors    = []

    for file in files:
        if not file or file.filename == "":
            continue
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{file.filename}: unsupported format")
            continue

        filename = secure_filename(file.filename)
        tmp_path = UPLOAD_DIR / filename
        file.save(tmp_path)

        try:
            text  = extract_text(tmp_path)
            terms = run_extraction(text)

            all_terms["named_entities"].extend(terms["named_entities"])
            all_terms["custom_patterns"].extend(terms["custom_patterns"])
            all_terms["sensitive_keywords"].extend(terms["sensitive_keywords"])

            processed.append({
                "filename": filename,
                "chars_extracted": len(text),
                "entities_found":  len(terms["named_entities"]),
                "patterns_found":  len(terms["custom_patterns"]),
                "keywords_found":  len(terms["sensitive_keywords"]),
            })
        finally:
            tmp_path.unlink(missing_ok=True)   # delete immediately

    profile = build_profile(
        all_terms,
        company_id=company_id,
        passphrase=passphrase or None,
    )

    return jsonify({
        "status":    "success",
        "processed": processed,
        "errors":    errors,
        "preview": {
            "entities":    [e["text"] for e in profile["detection_rules"]["exact_match"][:30]],
            "patterns":    profile["detection_rules"]["pattern_match"][:10],
            "keywords":    profile["detection_rules"]["keyword_match"][:30],
        },
        "stats": profile["stats"],
    })


@app.route("/profile/<company_id>", methods=["GET"])
def get_profile(company_id):
    path = PROFILE_DIR / f"{company_id}_profile.json"
    if not path.exists():
        return jsonify({"error": "Profile not found"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/profile/<company_id>/download", methods=["GET"])
def download_profile(company_id):
    return send_from_directory(PROFILE_DIR, f"{company_id}_profile.json", as_attachment=True)


@app.route("/profile/<company_id>/threshold", methods=["POST"])
def update_threshold(company_id):
    path = PROFILE_DIR / f"{company_id}_profile.json"
    if not path.exists():
        return jsonify({"error": "Profile not found"}), 404
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    profile["threshold"] = float(request.json.get("threshold", 0.82))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "updated", "threshold": profile["threshold"]})


@app.route("/profile/<company_id>/term", methods=["DELETE"])
def remove_term(company_id):
    path = PROFILE_DIR / f"{company_id}_profile.json"
    if not path.exists():
        return jsonify({"error": "Profile not found"}), 404
    term_to_remove = request.json.get("term", "").lower()
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    profile["detection_rules"]["exact_match"] = [
        e for e in profile["detection_rules"]["exact_match"]
        if e["text"].lower() != term_to_remove
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "removed"})


@app.route("/profile/<company_id>/term", methods=["POST"])
def add_term(company_id):
    path = PROFILE_DIR / f"{company_id}_profile.json"
    if not path.exists():
        return jsonify({"error": "Profile not found"}), 404
    new_term = request.json.get("term", "").strip()
    if not new_term:
        return jsonify({"error": "Empty term"}), 400
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    existing = [e["text"].lower() for e in profile["detection_rules"]["exact_match"]]
    if new_term.lower() not in existing:
        profile["detection_rules"]["exact_match"].append({"text": new_term, "type": "CUSTOM"})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "added"})


# ─── Inline Admin UI HTML ─────────────────────────────────────────────────────

ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DataShield Admin — Company Profile Generator</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --accent: #58a6ff; --danger: #f85149; --success: #3fb950;
    --text: #c9d1d9; --muted: #6e7681; --tag-bg: #1f2937;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
         font-size: 14px; min-height: 100vh; }
  .header { background: var(--surface); border-bottom: 1px solid var(--border);
            padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  .header .logo { font-size: 18px; font-weight: 700; color: var(--accent); }
  .header .sub { color: var(--muted); }
  .main { max-width: 960px; margin: 32px auto; padding: 0 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
          padding: 24px; margin-bottom: 24px; }
  .card h2 { font-size: 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  input, select { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                  color: var(--text); padding: 8px 12px; width: 100%; font-size: 13px; }
  input:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; gap: 12px; margin-bottom: 12px; }
  .row label { display: block; margin-bottom: 4px; color: var(--muted); font-size: 12px; }
  .col { flex: 1; }
  .drop-zone { border: 2px dashed var(--border); border-radius: 8px; padding: 40px 24px;
               text-align: center; cursor: pointer; transition: border-color 0.2s;
               color: var(--muted); }
  .drop-zone:hover, .drop-zone.dragover { border-color: var(--accent); color: var(--text); }
  .drop-zone input[type=file] { display: none; }
  .file-list { margin-top: 12px; }
  .file-item { background: var(--tag-bg); border-radius: 6px; padding: 8px 12px;
               margin-top: 6px; display: flex; justify-content: space-between;
               align-items: center; font-size: 12px; }
  .btn { background: var(--accent); color: #000; border: none; border-radius: 6px;
         padding: 10px 20px; font-weight: 600; cursor: pointer; font-size: 13px;
         transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn.danger { background: var(--danger); color: #fff; }
  .btn.secondary { background: var(--tag-bg); color: var(--text); border: 1px solid var(--border); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .tag { display: inline-block; background: var(--tag-bg); border: 1px solid var(--border);
         border-radius: 4px; padding: 3px 8px; font-size: 11px; margin: 3px;
         cursor: pointer; transition: all 0.15s; }
  .tag:hover { border-color: var(--danger); color: var(--danger); }
  .tag.removed { opacity: 0.3; text-decoration: line-through; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
  .stat { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
          padding: 12px; text-align: center; }
  .stat .num { font-size: 28px; font-weight: 700; color: var(--accent); }
  .stat .lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .alert { padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; }
  .alert.success { background: rgba(63,185,80,.15); border: 1px solid var(--success);
                   color: var(--success); }
  .alert.error   { background: rgba(248,81,73,.15); border: 1px solid var(--danger);
                   color: var(--danger); }
  .slider-row { display: flex; align-items: center; gap: 12px; }
  input[type=range] { flex: 1; accent-color: var(--accent); }
  .threshold-val { font-size: 16px; font-weight: 700; color: var(--accent); min-width: 40px; }
  .loading { display: none; color: var(--muted); font-size: 13px; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border);
             border-top-color: var(--accent); border-radius: 50%;
             animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="header">
  <span class="logo">🛡 DataShield</span>
  <span class="sub">/ Admin — Company Profile Generator</span>
</div>
<div class="main">

  <div class="card" id="upload-card">
    <h2>📂 Upload Sensitive Company Documents</h2>
    <p style="color:var(--muted);margin-bottom:16px;font-size:13px">
      Documents are processed locally. No data leaves this machine.
      Supported: PDF, DOCX, TXT, XLSX, CSV
    </p>
    <div class="row">
      <div class="col">
        <label>Company ID (alphanumeric)</label>
        <input type="text" id="company-id" placeholder="e.g. acme_corp" value="my_company">
      </div>
      <div class="col">
        <label>Encryption Passphrase (optional)</label>
        <input type="password" id="passphrase" placeholder="Leave blank to skip encryption">
      </div>
    </div>
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
      <input type="file" id="file-input" multiple accept=".pdf,.docx,.txt,.xlsx,.csv">
      <div>⬆ Drop files here or click to browse</div>
      <div style="font-size:11px;margin-top:6px">Supports PDF, DOCX, TXT, XLSX, CSV</div>
    </div>
    <div class="file-list" id="file-list"></div>
    <div style="margin-top:16px;display:flex;gap:12px;align-items:center">
      <button class="btn" id="process-btn" onclick="processFiles()" disabled>
        ⚙ Extract & Build Profile
      </button>
      <span class="loading" id="loading-msg">
        <span class="spinner"></span>Processing documents…
      </span>
    </div>
  </div>

  <div id="results-card" class="card" style="display:none">
    <h2>✅ Extracted Terms</h2>
    <div id="alert-area"></div>
    <div class="stats" id="stats-area"></div>
    <div style="margin-top:20px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-weight:600">Named Entities & Codes</span>
        <span style="font-size:11px;color:var(--muted)">Click to remove</span>
      </div>
      <div id="tags-area"></div>
    </div>
    <div style="margin-top:20px">
      <label style="color:var(--muted);font-size:12px;display:block;margin-bottom:6px">
        Add Custom Term
      </label>
      <div style="display:flex;gap:8px">
        <input type="text" id="new-term" placeholder="e.g. Project Phoenix">
        <button class="btn secondary" onclick="addTerm()" style="white-space:nowrap">+ Add</button>
      </div>
    </div>
    <div style="margin-top:20px">
      <div style="margin-bottom:8px;font-weight:600">Sensitivity Threshold</div>
      <div class="slider-row">
        <span style="color:var(--muted);font-size:12px">Permissive</span>
        <input type="range" id="threshold" min="0.5" max="0.99" step="0.01" value="0.82"
               oninput="document.getElementById('tval').textContent=this.value">
        <span style="color:var(--muted);font-size:12px">Strict</span>
        <span class="threshold-val" id="tval">0.82</span>
      </div>
    </div>
    <div style="margin-top:20px;display:flex;gap:12px">
      <button class="btn" onclick="applyProfile()">💾 Apply & Export Profile</button>
      <button class="btn secondary" onclick="downloadProfile()">⬇ Download profile.json</button>
    </div>
  </div>

</div>

<script>
let selectedFiles = [];
let currentCompanyId = 'my_company';

// ── Drag and drop ──
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  addFiles(e.dataTransfer.files);
});
document.getElementById('file-input').addEventListener('change', e => addFiles(e.target.files));

function addFiles(fileList) {
  for (const f of fileList) selectedFiles.push(f);
  renderFileList();
  document.getElementById('process-btn').disabled = selectedFiles.length === 0;
}

function renderFileList() {
  const el = document.getElementById('file-list');
  el.innerHTML = selectedFiles.map((f, i) => `
    <div class="file-item">
      <span>📄 ${f.name} <span style="color:var(--muted)">(${(f.size/1024).toFixed(0)} KB)</span></span>
      <button class="btn danger" style="padding:3px 10px;font-size:11px" onclick="removeFile(${i})">✕</button>
    </div>`).join('');
}

function removeFile(i) {
  selectedFiles.splice(i, 1);
  renderFileList();
  document.getElementById('process-btn').disabled = selectedFiles.length === 0;
}

// ── Upload & Process ──
async function processFiles() {
  currentCompanyId = document.getElementById('company-id').value.trim() || 'company';
  const formData = new FormData();
  formData.append('company_id', currentCompanyId);
  formData.append('passphrase', document.getElementById('passphrase').value);
  for (const f of selectedFiles) formData.append('documents', f);

  document.getElementById('loading-msg').style.display = 'inline';
  document.getElementById('process-btn').disabled = true;

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    document.getElementById('loading-msg').style.display = 'none';
    document.getElementById('process-btn').disabled = false;

    if (data.status === 'success') {
      renderResults(data);
      document.getElementById('results-card').style.display = 'block';
      document.getElementById('results-card').scrollIntoView({ behavior: 'smooth' });
    } else {
      showAlert('error', 'Processing failed: ' + JSON.stringify(data));
    }
  } catch (err) {
    document.getElementById('loading-msg').style.display = 'none';
    document.getElementById('process-btn').disabled = false;
    showAlert('error', 'Request failed: ' + err.message);
  }
}

function renderResults(data) {
  // Stats
  document.getElementById('stats-area').innerHTML = `
    <div class="stat"><div class="num">${data.stats.exact_terms}</div><div class="lbl">Named Entities</div></div>
    <div class="stat"><div class="num">${data.stats.regex_patterns}</div><div class="lbl">Code Patterns</div></div>
    <div class="stat"><div class="num">${data.stats.keywords}</div><div class="lbl">Keywords</div></div>
  `;
  // Tags
  document.getElementById('tags-area').innerHTML =
    data.preview.entities.map(e =>
      `<span class="tag" onclick="toggleTag(this,'${e.replace(/'/g,"\\'")}')">${e}</span>`
    ).join('');
  // Alert
  const errs = data.errors || [];
  showAlert('success',
    `Processed ${data.processed.length} file(s). ${errs.length > 0 ? 'Errors: ' + errs.join(', ') : ''}`
  );
}

function toggleTag(el, term) {
  el.classList.toggle('removed');
  const action = el.classList.contains('removed') ? 'DELETE' : 'POST';
  fetch(`/profile/${currentCompanyId}/term`, {
    method: action,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ term }),
  });
}

async function addTerm() {
  const input = document.getElementById('new-term');
  const term  = input.value.trim();
  if (!term) return;
  const resp = await fetch(`/profile/${currentCompanyId}/term`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ term }),
  });
  if (resp.ok) {
    const el = document.createElement('span');
    el.className = 'tag';
    el.textContent = term;
    el.onclick = () => toggleTag(el, term);
    document.getElementById('tags-area').appendChild(el);
    input.value = '';
  }
}

async function applyProfile() {
  const threshold = parseFloat(document.getElementById('threshold').value);
  await fetch(`/profile/${currentCompanyId}/threshold`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold }),
  });
  showAlert('success', `Profile applied with threshold ${threshold}. Ready for deployment.`);
}

function downloadProfile() {
  window.open(`/profile/${currentCompanyId}/download`);
}

function showAlert(type, msg) {
  document.getElementById('alert-area').innerHTML =
    `<div class="alert ${type}">${msg}</div>`;
}
</script>
</body>
</html>"""

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8889
    print(f"\n🛡 DataShield Admin Panel")
    print(f"   URL: http://127.0.0.1:{port}")
    print(f"   Profiles saved to: {PROFILE_DIR.resolve()}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host="127.0.0.1", port=port, debug=False)
