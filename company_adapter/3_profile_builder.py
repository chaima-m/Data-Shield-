"""
DataShield AI — Step 3: Profile Builder
=========================================
Combines enriched terms + embeddings into the final company_profile.json.

The output format matches exactly what CompanyRuleEngine in
4_detection_engine.js expects — so you can drop the JSON into
chrome.storage.local and the extension immediately picks it up.

Output structure:
  detection_rules.exact_match    → fed to CompanyRuleEngine.exactTerms
  detection_rules.pattern_match  → fed to CompanyRuleEngine.regexPatterns
  detection_rules.keyword_match  → fed to CompanyRuleEngine.keywordSet
  semantic_index                 → used by the lightweight JS similarity check

Input:  ./adapter_output/enriched_terms.json
Output: ./adapter_output/company_profile.json
        ./adapter_output/company_profile_lite.json  (no vectors, for debugging)

Usage:
    python company_adapter/3_profile_builder.py
    python company_adapter/3_profile_builder.py --company_id "acme_corp" --threshold 0.82
"""

import re
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from adapter_utils import load_json, save_json

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_FILE         = "./adapter_output/enriched_terms.json"
OUTPUT_FILE        = "./adapter_output/company_profile.json"
OUTPUT_LITE_FILE   = "./adapter_output/company_profile_lite.json"

# Label priority (higher → preferred when conflict)
LABEL_PRIORITY = {
    "credentials":  6,
    "pii":          5,
    "financial":    4,
    "health":       3,
    "confidential": 2,
    "safe":         0,
}

# Entity types → detection label mapping
ENTITY_TO_LABEL = {
    "ORG":        "confidential",
    "PRODUCT":    "confidential",
    "WORK_OF_ART": "confidential",
    "EVENT":      "confidential",
    "FAC":        "confidential",
    "LAW":        "confidential",
    "GPE":        "confidential",
    "KEYWORD":    "confidential",
    "BIGRAM":     "confidential",
    "CODE":       "confidential",
}

# Minimum score for a term to enter exact_match (higher bar)
MIN_SCORE_EXACT   = 0.55
# Minimum score for keyword_match (lower bar, broader coverage)
MIN_SCORE_KEYWORD = 0.25
# How many representative semantic vectors to store in the profile
# (used by the JS semantic similarity layer)
MAX_SEMANTIC_VECTORS = 80


# ─── Rule Builders ────────────────────────────────────────────────────────────

def build_exact_match_rules(terms: list[dict]) -> list[dict]:
    """
    Build the exact_match list:
    High-confidence multi-character terms matched literally (case-insensitive).
    """
    rules = []
    seen  = set()
    for t in terms:
        if t.get("score", 0) < MIN_SCORE_EXACT:
            continue
        text = t["text"].strip()
        key  = text.lower()
        if key in seen or len(key) < 4:
            continue
        seen.add(key)
        rules.append({
            "text":   text,
            "type":   t.get("type", "UNKNOWN"),
            "source": t.get("source", "extracted"),
            "score":  t.get("score", 0),
        })
    # Sort by score descending
    rules.sort(key=lambda r: -r.get("score", 0))
    return rules


def build_pattern_match_rules(code_patterns: list[dict]) -> list[dict]:
    """
    Build the pattern_match list from code pattern extraction.
    These are compiled as RegExp objects in the JS engine.
    """
    rules = []
    seen  = set()
    for cp in code_patterns:
        pattern = cp.get("pattern", "")
        if not pattern or pattern in seen:
            continue
        # Validate the regex is valid Python (and likely valid JS too)
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error:
            continue
        seen.add(pattern)
        rules.append({
            "pattern": pattern,
            "label":   cp.get("label", "confidential"),
            "original": cp.get("original", ""),
        })
    return rules


def build_keyword_match_list(terms: list[dict]) -> list[str]:
    """
    Build the keyword_match list:
    Single-word and short keywords for density-based detection.
    (≥2 keywords in a prompt → flagged)
    """
    keywords = set()
    for t in terms:
        if t.get("score", 0) < MIN_SCORE_KEYWORD:
            continue
        text = t["text"].strip().lower()
        # Add the full term as a keyword
        keywords.add(text)
        # If multi-word, also add individual significant words
        words = text.split()
        if len(words) > 1:
            for word in words:
                if len(word) >= 5 and word not in _STOPWORDS_MINI:
                    keywords.add(word)
    return sorted(keywords)


# Small inline stopword set (duplicates adapter_utils for standalone use)
_STOPWORDS_MINI = {
    "the", "and", "for", "that", "this", "with", "from", "have", "will",
    "been", "they", "their", "also", "which", "when", "where", "about",
    "some", "than", "other", "into", "more", "these", "those", "each",
    "such", "after", "before", "over", "under", "through", "company",
    "document", "page", "section", "information", "please", "date",
}


def build_semantic_index(terms: list[dict], max_vectors: int = MAX_SEMANTIC_VECTORS) -> dict:
    """
    Build the semantic_index for the browser JS similarity layer.

    Stores the top-N most important term vectors (64-dim PCA-compressed).
    The browser JS loads these and computes cosine similarity against
    an incoming prompt's lightweight representation.

    Format:
      {
        "terms":   ["Project Helios", "NovaTech", ...],
        "labels":  ["confidential", "confidential", ...],
        "vectors": [[0.12, -0.34, ...], ...]   (64-dim each)
      }
    """
    # Sort by score and take top N
    scored = sorted(
        [t for t in terms if "vector" in t],
        key=lambda t: -t.get("score", 0)
    )[:max_vectors]

    return {
        "terms":   [t["text"] for t in scored],
        "labels":  [ENTITY_TO_LABEL.get(t.get("type", "KEYWORD"), "confidential") for t in scored],
        "vectors": [t["vector"] for t in scored],  # already 64-dim from Step 2
    }


# ─── Profile Assembly ─────────────────────────────────────────────────────────

def build_profile(
    enriched: dict,
    company_id: str   = "my_company",
    threshold: float  = 0.82,
) -> dict:
    """
    Assemble the final profile.json from enriched terms.
    """
    terms         = enriched.get("terms", [])
    code_patterns = enriched.get("code_patterns", [])
    pca           = enriched.get("pca", {})

    print(f"\n Building detection rules from {len(terms)} enriched terms...")

    exact_match   = build_exact_match_rules(terms)
    pattern_match = build_pattern_match_rules(code_patterns)
    keyword_match = build_keyword_match_list(terms)
    semantic_idx  = build_semantic_index(terms)

    print(f"   exact_match rules:   {len(exact_match)}")
    print(f"   pattern_match rules: {len(pattern_match)}")
    print(f"   keyword_match items: {len(keyword_match)}")
    print(f"   semantic vectors:    {len(semantic_idx['terms'])}")

    profile = {
        "company_id":   company_id,
        "version":      "1.0",
        "created_at":   datetime.utcnow().isoformat() + "Z",
        "threshold":    threshold,

        # ── Detection rules (consumed by CompanyRuleEngine in 4_detection_engine.js)
        "detection_rules": {
            "exact_match":   exact_match,
            "pattern_match": pattern_match,
            "keyword_match": keyword_match,
        },

        # ── Semantic index (consumed by SemanticLayer in 5_company_adapter.js)
        "semantic_index": {
            "embedding_model": enriched.get("embedding_model", ""),
            "pca_n_components": pca.get("n_components", PCA_DIMS_DEFAULT),
            "pca_components":   pca.get("components", []),   # (64, 384)
            "pca_mean":         pca.get("mean", []),         # (384,)
            "threshold":        threshold,
            **semantic_idx,
        },

        # ── Metadata
        "metadata": {
            "total_terms":       len(terms),
            "exact_rules":       len(exact_match),
            "pattern_rules":     len(pattern_match),
            "keyword_count":     len(keyword_match),
            "semantic_vectors":  len(semantic_idx["terms"]),
            "embedding_model":   enriched.get("embedding_model", ""),
            "stats":             enriched.get("stats", {}),
        },
    }

    return profile


PCA_DIMS_DEFAULT = 64


# ─── Lite Profile (no vectors, for debugging/inspection) ─────────────────────

def build_lite_profile(profile: dict) -> dict:
    """
    Strip out the large vector arrays for a human-readable version.
    Useful for debugging and manual review.
    """
    import copy
    lite = copy.deepcopy(profile)

    # Remove vectors from detection_rules exact_match (they're small anyway)
    # Remove the PCA matrix and term vectors from semantic_index
    si = lite.get("semantic_index", {})
    si.pop("pca_components", None)
    si.pop("pca_mean", None)
    si.pop("vectors", None)

    # Summarize instead
    si["_note"] = "Vectors removed in lite version. See company_profile.json for full data."

    return lite


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataShield: Build company_profile.json from enriched terms"
    )
    parser.add_argument("--company_id", type=str, default="my_company",
                        help="Company identifier (no spaces)")
    parser.add_argument("--threshold",  type=float, default=0.82,
                        help="Semantic similarity threshold (0.0–1.0, default 0.82)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DataShield AI — Step 3: Profile Builder")
    print("=" * 60)

    enriched = load_json(INPUT_FILE)
    if enriched is None:
        print(f"\n[ERROR] Could not load {INPUT_FILE}")
        print("  Run Step 2 first: python company_adapter/2_build_embeddings.py")
        sys.exit(1)

    print(f"\n Loaded enriched terms from: {INPUT_FILE}")

    # Build full profile
    profile = build_profile(
        enriched,
        company_id=args.company_id,
        threshold=args.threshold,
    )

    # Save full profile (with vectors)
    save_json(profile, OUTPUT_FILE)

    # Save lite profile (without vectors, for review)
    lite = build_lite_profile(profile)
    save_json(lite, OUTPUT_LITE_FILE)

    # ── Print summary ─────────────────────────────────────────────────────────
    m = profile["metadata"]
    print("\n" + "=" * 60)
    print(f"  Profile built for: {args.company_id}")
    print(f"  Threshold:         {args.threshold}")
    print(f"  Exact match rules: {m['exact_rules']}")
    print(f"  Pattern rules:     {m['pattern_rules']}")
    print(f"  Keywords:          {m['keyword_count']}")
    print(f"  Semantic vectors:  {m['semantic_vectors']}")
    print("=" * 60)

    # Estimate file size
    full_size = os.path.getsize(OUTPUT_FILE) / 1024
    lite_size = os.path.getsize(OUTPUT_LITE_FILE) / 1024
    print(f"\n  company_profile.json      : {full_size:.1f} KB (deploy this to extension)")
    print(f"  company_profile_lite.json : {lite_size:.1f} KB (human-readable, for review)")

    print("\n Next step: python company_adapter/4_test_adapter.py")
    print("\n Deploy instructions:")
    print("  1. Copy adapter_output/company_profile.json to extension")
    print("  2. Load via: chrome.storage.local.set({companyProfile: <json_string>})")
    print("  3. OR: deploy via Group Policy / IT script (see README)")
