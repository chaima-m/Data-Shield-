"""
DataShield AI — Step 1: Term Extraction
========================================
Extracts sensitive terms, entities, and patterns from company documents using:
  - spaCy multilingual NER (named entity recognition)
  - Internal code pattern matching (regex)
  - High-frequency domain noun extraction (TF-IDF style)

Input:  A folder of company documents (PDF, DOCX, TXT, XLSX, CSV)
Output: ./adapter_output/extracted_terms.json

Usage (Colab):
    !python company_adapter/1_extract_terms.py --docs_dir /path/to/your/docs
    # or use the default sample_docs/ folder for testing
"""

import re
import os
import sys
import argparse
from pathlib import Path
from collections import Counter

from tqdm import tqdm
from adapter_utils import (
    clean_text, extract_text_from_file, split_into_sentences,
    find_internal_codes, is_meaningful_term, save_json, STOPWORDS
)

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR      = "./adapter_output"
OUTPUT_FILE     = f"{OUTPUT_DIR}/extracted_terms.json"

# NER entity types to treat as potentially sensitive
SENSITIVE_ENTITY_TYPES = {
    "ORG",          # Organizations, companies, subsidiaries
    "PRODUCT",      # Product names, internal tools
    "WORK_OF_ART",  # Project names, codenames
    "LAW",          # Regulations, internal policies
    "GPE",          # Geopolitical — offices, regions (medium risk)
    "FAC",          # Facilities — data centers, offices
    "EVENT",        # Events, launches, operations
    "LANGUAGE",     # Custom (in some models maps to project codes)
}

# Minimum frequency for a term to be kept as a keyword
MIN_KEYWORD_FREQ = 3
# Minimum entity frequency to include
MIN_ENTITY_FREQ  = 1
# Max terms per category (to keep profile light)
MAX_EXACT_TERMS  = 300
MAX_KEYWORDS     = 200
MAX_CODE_PATTERNS = 100

# ─── NER Loader ──────────────────────────────────────────────────────────────

def load_spacy_model():
    """
    Load the small multilingual spaCy model.
    Automatically downloads it if not present.
    Supports: EN, FR, AR (partial) + 40 other languages.
    """
    import spacy
    model_name = "xx_ent_wiki_sm"
    try:
        nlp = spacy.load(model_name)
        print(f"  spaCy model '{model_name}' loaded.")
    except OSError:
        print(f"  Downloading spaCy model '{model_name}'...")
        os.system(f"python -m spacy download {model_name}")
        import spacy
        nlp = spacy.load(model_name)
    # Disable unused pipeline components for speed
    disabled = [p for p in nlp.pipe_names if p not in ("ner", "tok2vec")]
    nlp.select_pipes(disable=disabled)
    return nlp


# ─── Entity Extraction ────────────────────────────────────────────────────────

def extract_entities_from_text(text: str, nlp, batch_size: int = 50) -> list[dict]:
    """
    Run spaCy NER over the text in chunks.
    Returns list of { text, label, count } dicts.
    """
    sentences = split_into_sentences(text, max_chars=300)
    entity_counter = Counter()
    entity_labels  = {}

    # Process in batches for speed
    docs = nlp.pipe(sentences, batch_size=batch_size)
    for doc in docs:
        for ent in doc.ents:
            t = ent.text.strip()
            if len(t) < 3 or not is_meaningful_term(t, min_len=3):
                continue
            if ent.label_ not in SENSITIVE_ENTITY_TYPES:
                continue
            key = t.lower()
            entity_counter[key] += 1
            # Keep the most common casing
            if key not in entity_labels:
                entity_labels[key] = {"text": t, "type": ent.label_}

    results = []
    for key, count in entity_counter.most_common():
        if count < MIN_ENTITY_FREQ:
            continue
        results.append({
            "text":  entity_labels[key]["text"],
            "type":  entity_labels[key]["type"],
            "count": count,
        })
    return results[:MAX_EXACT_TERMS]


# ─── Keyword Frequency Extraction ────────────────────────────────────────────

def extract_domain_keywords(text: str) -> list[dict]:
    """
    Extract high-frequency multi-word nouns and domain-specific terms.
    Uses a simple frequency approach (TF-IDF proxy).
    """
    # Tokenize: keep words and hyphenated compounds
    tokens = re.findall(r"\b[\w][\w'-]{3,}\b", text)
    # Lowercase for counting
    lower_tokens = [t.lower() for t in tokens]
    freq = Counter(lower_tokens)

    keywords = []
    for word, count in freq.most_common(500):
        if count < MIN_KEYWORD_FREQ:
            break
        if not is_meaningful_term(word, min_len=5):
            continue
        # Prefer domain-like terms (mixed case in source, acronyms, etc.)
        keywords.append({"text": word, "count": count})
    return keywords[:MAX_KEYWORDS]


# ─── Bigram Extraction (2-word Phrases) ──────────────────────────────────────

def extract_bigrams(text: str) -> list[dict]:
    """
    Extract frequent 2-word phrases that might be project/client names.
    e.g. 'Project Helios', 'Operation Sunrise', 'NovaTech client'
    """
    words = re.findall(r"\b[A-Za-z\u0600-\u06FF]{2,}\b", text)
    bigram_counter = Counter()
    for i in range(len(words) - 1):
        w1, w2 = words[i].lower(), words[i + 1].lower()
        if w1 in STOPWORDS or w2 in STOPWORDS:
            continue
        if len(w1) < 3 or len(w2) < 3:
            continue
        bigram_counter[(words[i], words[i + 1])] += 1

    results = []
    for (w1, w2), count in bigram_counter.most_common(100):
        if count < 2:
            break
        phrase = f"{w1} {w2}"
        results.append({"text": phrase, "count": count})
    return results


# ─── Code Pattern Extraction ──────────────────────────────────────────────────

def extract_code_patterns(text: str) -> list[dict]:
    """
    Find internal codes and generate regex patterns for them.
    e.g. 'PROJECT_HELIOS' → regex: 'PROJECT[-_]?HELIOS'
    """
    codes = find_internal_codes(text)
    results = []
    seen_patterns = set()

    for code in codes[:MAX_CODE_PATTERNS]:
        # Generate a regex that matches the code with slight variations
        pattern = _code_to_regex(code)
        if pattern and pattern not in seen_patterns:
            seen_patterns.add(pattern)
            results.append({
                "original": code,
                "pattern":  pattern,
                "label":    "confidential",
            })
    return results


def _code_to_regex(code: str) -> str:
    """
    Convert a detected internal code string into a flexible regex.
    Examples:
      'PROJECT-HELIOS' → r'PROJECT[-_\s]?HELIOS'
      'CLIENT_XK2'     → r'CLIENT[-_\s]?XK2'
      'ProjectHelios'  → r'Project\s*Helios'
    """
    # CamelCase → insert optional whitespace between words
    camel = re.sub(r"(?<=[a-z])(?=[A-Z])", r"\\s*", code)
    if camel != code:
        return re.escape(camel).replace("\\\\s\\*", r"\s*")

    # UPPERCASE_UNDERSCORE or UPPERCASE-DASH
    if re.fullmatch(r"[A-Z0-9]+[-_][A-Z0-9]+", code):
        parts = re.split(r"[-_]", code)
        return r"[-_\s]?".join(re.escape(p) for p in parts)

    # Default: just escape the whole thing
    return re.escape(code)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def extract_from_folder(docs_dir: str) -> dict:
    """
    Run the full extraction pipeline over all documents in a folder.
    Returns the combined extracted_terms dict.
    """
    docs_dir = Path(docs_dir)
    supported_exts = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".xls", ".csv"}
    files = [f for f in docs_dir.rglob("*") if f.suffix.lower() in supported_exts]

    if not files:
        print(f"\n[ERROR] No supported files found in '{docs_dir}'.")
        print(f"  Supported types: {', '.join(supported_exts)}")
        sys.exit(1)

    print(f"\n Found {len(files)} document(s) in '{docs_dir}'")

    # Load spaCy
    print("\n Loading spaCy NER model...")
    nlp = load_spacy_model()

    all_entities   = Counter()
    entity_meta    = {}
    all_keywords   = Counter()
    all_bigrams    = Counter()
    all_codes      = []
    processed_files = []

    for fpath in tqdm(files, desc=" Extracting", unit="file"):
        fname = fpath.name
        print(f"\n  Processing: {fname}")

        raw_text = extract_text_from_file(str(fpath))
        if not raw_text.strip():
            print(f"    → Empty or unreadable, skipping.")
            continue

        text = clean_text(raw_text)
        char_count = len(text)
        print(f"    → {char_count:,} chars extracted")

        # 1. Named entities
        entities = extract_entities_from_text(text, nlp)
        for e in entities:
            key = e["text"].lower()
            all_entities[key] = all_entities.get(key, 0) + e["count"]
            if key not in entity_meta:
                entity_meta[key] = {"text": e["text"], "type": e["type"]}
        print(f"    → {len(entities)} entities found")

        # 2. Domain keywords
        keywords = extract_domain_keywords(text)
        for kw in keywords:
            all_keywords[kw["text"]] += kw["count"]
        print(f"    → {len(keywords)} keywords found")

        # 3. Bigrams
        bigrams = extract_bigrams(text)
        for bg in bigrams:
            all_bigrams[bg["text"]] += bg["count"]
        print(f"    → {len(bigrams)} bigrams found")

        # 4. Internal codes
        codes = extract_code_patterns(text)
        all_codes.extend(codes)
        print(f"    → {len(codes)} code patterns found")

        processed_files.append({
            "name":     fname,
            "chars":    char_count,
            "entities": len(entities),
            "keywords": len(keywords),
        })

    # ── Aggregate & Rank ──────────────────────────────────────────────────────
    final_entities = []
    for key, count in all_entities.most_common(MAX_EXACT_TERMS):
        final_entities.append({
            "text":  entity_meta[key]["text"],
            "type":  entity_meta[key]["type"],
            "count": count,
        })

    final_keywords = [
        {"text": word, "count": count}
        for word, count in all_keywords.most_common(MAX_KEYWORDS)
        if is_meaningful_term(word)
    ]

    final_bigrams = [
        {"text": bg, "count": count}
        for bg, count in all_bigrams.most_common(100)
        if count >= 2
    ]

    # Deduplicate codes
    seen = set()
    final_codes = []
    for c in all_codes:
        if c["pattern"] not in seen:
            seen.add(c["pattern"])
            final_codes.append(c)

    result = {
        "version":        "1.0",
        "docs_dir":       str(docs_dir),
        "files_processed": processed_files,
        "entities":       final_entities,
        "keywords":       final_keywords,
        "bigrams":        final_bigrams,
        "code_patterns":  final_codes,
    }

    print(f"\n Extraction complete:")
    print(f"   Entities:     {len(final_entities)}")
    print(f"   Keywords:     {len(final_keywords)}")
    print(f"   Bigrams:      {len(final_bigrams)}")
    print(f"   Code patterns:{len(final_codes)}")

    return result


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataShield: Extract terms from company documents")
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="./company_adapter/sample_docs",
        help="Folder containing company documents to process"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DataShield AI — Step 1: Term Extraction")
    print("=" * 60)

    terms = extract_from_folder(args.docs_dir)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(terms, OUTPUT_FILE)

    print(f"\n Done. Output saved to: {OUTPUT_FILE}")
    print(" Next step: python company_adapter/2_build_embeddings.py")
