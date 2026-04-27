"""
DataShield AI — Step 4: Test the Company Adapter
=================================================
Loads company_profile.json and simulates the three detection layers
exactly as the browser extension does (pattern + rule + semantic).

Tests your profile against:
  - Known sensitive prompts (should be flagged)
  - Safe prompts (should pass)
  - Multilingual prompts (EN/FR/AR)
  - Edge cases

Prints a detailed detection report with per-layer breakdown.

Usage:
    python company_adapter/4_test_adapter.py
    python company_adapter/4_test_adapter.py --profile ./adapter_output/company_profile.json
"""

import re
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path

from adapter_utils import load_json

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_PROFILE = "./adapter_output/company_profile.json"

# Cosine similarity threshold for semantic layer
SEMANTIC_THRESHOLD = 0.82

# Keyword density threshold (how many keywords must appear)
KEYWORD_DENSITY_THRESHOLD = 2


# ─── Python Replica of CompanyRuleEngine (mirrors 4_detection_engine.js) ─────

class CompanyRuleEngine:
    """
    Python replica of the JS CompanyRuleEngine in 4_detection_engine.js.
    Used to test the profile before deploying to the extension.
    """
    def __init__(self, profile: dict):
        rules = profile.get("detection_rules", {})

        self.exact_terms   = [
            e["text"].lower().strip()
            for e in rules.get("exact_match", [])
            if len(e.get("text", "")) > 3
        ]

        self.regex_patterns = []
        for p in rules.get("pattern_match", []):
            try:
                self.regex_patterns.append(
                    re.compile(p["pattern"], re.IGNORECASE)
                )
            except re.error:
                pass

        self.keyword_set = set(rules.get("keyword_match", []))

        print(f"  [RuleEngine] Loaded: {len(self.exact_terms)} exact terms, "
              f"{len(self.regex_patterns)} patterns, {len(self.keyword_set)} keywords")

    def analyze(self, text: str) -> dict:
        lower = text.lower()

        # 1. Exact match
        for term in self.exact_terms:
            if term in lower:
                return {
                    "matched":  True,
                    "layer":    "exact_match",
                    "term":     term,
                    "label":    "confidential",
                    "severity": "high",
                }

        # 2. Regex pattern match
        for pattern in self.regex_patterns:
            m = pattern.search(text)
            if m:
                return {
                    "matched":  True,
                    "layer":    "pattern_match",
                    "term":     m.group(0),
                    "label":    "confidential",
                    "severity": "high",
                }

        # 3. Keyword density
        words = lower.split()
        hits  = [w for w in words if w in self.keyword_set]
        if len(hits) >= KEYWORD_DENSITY_THRESHOLD:
            return {
                "matched":  True,
                "layer":    "keyword_density",
                "term":     ", ".join(set(hits[:5])),
                "label":    "confidential",
                "severity": "medium",
            }

        return {"matched": False}


# ─── Python Replica of SemanticLayer (mirrors 5_company_adapter.js) ──────────

class SemanticLayer:
    """
    Python replica of the JS SemanticLayer in 5_company_adapter.js.
    Computes cosine similarity between incoming text and stored term vectors.

    Uses the same PCA projection as the JS layer for identical results.
    """
    def __init__(self, profile: dict):
        si = profile.get("semantic_index", {})

        self.threshold = si.get("threshold", SEMANTIC_THRESHOLD)
        self.terms     = si.get("terms", [])
        self.labels    = si.get("labels", [])
        self.vectors   = np.array(si.get("vectors", []), dtype=np.float32)

        # PCA params for projecting query embeddings
        self.pca_components = np.array(si.get("pca_components", []), dtype=np.float32)
        self.pca_mean       = np.array(si.get("pca_mean", []), dtype=np.float32)

        self._model = None
        print(f"  [SemanticLayer] Loaded: {len(self.terms)} semantic vectors")

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = "paraphrase-multilingual-MiniLM-L12-v2"
                print(f"  [SemanticLayer] Loading embedding model: {model_name}...")
                self._model = SentenceTransformer(model_name)
            except ImportError:
                print("  [SemanticLayer] sentence-transformers not available. Skipping semantic layer.")
                self._model = "unavailable"
        return self._model

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed the query text and project into the PCA space."""
        model = self._load_model()
        if model == "unavailable" or model is None:
            return None

        # Full embedding (384-dim)
        raw = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]

        # Project into PCA space
        if self.pca_components.size > 0 and self.pca_mean.size > 0:
            centered = raw - self.pca_mean
            projected = self.pca_components.dot(centered)
            # Normalize
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected = projected / norm
            return projected.astype(np.float32)
        return raw.astype(np.float32)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def analyze(self, text: str) -> dict:
        if len(self.vectors) == 0:
            return {"matched": False, "source": "semantic", "reason": "no_vectors"}

        query_vec = self._embed_query(text)
        if query_vec is None:
            return {"matched": False, "source": "semantic", "reason": "model_unavailable"}

        # Compute cosine similarity against all stored vectors
        best_score = 0.0
        best_term  = ""
        best_label = "confidential"

        for i, vec in enumerate(self.vectors):
            score = self._cosine(query_vec, vec)
            if score > best_score:
                best_score = score
                best_term  = self.terms[i] if i < len(self.terms) else ""
                best_label = self.labels[i] if i < len(self.labels) else "confidential"

        if best_score >= self.threshold:
            return {
                "matched":    True,
                "source":     "semantic",
                "term":       best_term,
                "similarity": round(best_score, 3),
                "label":      best_label,
                "severity":   "high" if best_score > 0.90 else "medium",
            }

        return {
            "matched":     False,
            "source":      "semantic",
            "best_score":  round(best_score, 3),
            "best_term":   best_term,
        }


# ─── Test Suite ───────────────────────────────────────────────────────────────

# Default test cases — will be SUPPLEMENTED with profile-specific tests at runtime
BASE_TEST_CASES = [
    # ── Should be caught by pattern/keyword/semantic ──────────────────────────
    {
        "id":       "SAFE_EN_1",
        "text":     "Can you summarize the main points of agile development methodology?",
        "expected": False,
        "lang":     "en",
        "note":     "Generic safe query",
    },
    {
        "id":       "SAFE_FR_1",
        "text":     "Comment fonctionne le machine learning? Explique-moi les bases.",
        "expected": False,
        "lang":     "fr",
        "note":     "Safe French query",
    },
    {
        "id":       "SAFE_AR_1",
        "text":     "ما هي أفضل الممارسات في تطوير البرمجيات؟",
        "expected": False,
        "lang":     "ar",
        "note":     "Safe Arabic query",
    },
    {
        "id":       "EDGE_SHORT",
        "text":     "hi",
        "expected": False,
        "lang":     "en",
        "note":     "Too short to be sensitive",
    },
]


def generate_profile_specific_tests(profile: dict) -> list[dict]:
    """
    Auto-generate test cases from the profile's own terms.
    This tests whether the profile actually catches what it should.
    """
    tests = []
    rules = profile.get("detection_rules", {})

    # Test a few exact match terms
    for i, term in enumerate(rules.get("exact_match", [])[:5]):
        text = term["text"]
        tests.append({
            "id":       f"EXACT_{i+1}",
            "text":     f"I need to share information about {text} with the team. Here are the details.",
            "expected": True,
            "lang":     "en",
            "note":     f"Should catch exact term: '{text}'",
        })

    # Test a pattern match
    for i, p in enumerate(rules.get("pattern_match", [])[:3]):
        tests.append({
            "id":       f"PATTERN_{i+1}",
            "text":     f"Reference code {p['original']} — please review this document.",
            "expected": True,
            "lang":     "en",
            "note":     f"Should match pattern: {p['pattern']}",
        })

    # Test keyword density
    keywords = list(rules.get("keyword_match", []))[:4]
    if len(keywords) >= 2:
        kw_text = " and ".join(keywords[:3])
        tests.append({
            "id":       "KEYWORD_DENSITY",
            "text":     f"This document discusses {kw_text}.",
            "expected": True,
            "lang":     "en",
            "note":     "Should trigger keyword density (≥2 keywords)",
        })

    return tests


# ─── Detection Runner ─────────────────────────────────────────────────────────

def run_detection(text: str, rule_engine: CompanyRuleEngine,
                  semantic_layer: SemanticLayer) -> dict:
    """Run both layers and return combined result."""
    rule_result     = rule_engine.analyze(text)
    semantic_result = semantic_layer.analyze(text)

    is_sensitive = rule_result.get("matched") or semantic_result.get("matched")

    triggered = []
    if rule_result.get("matched"):
        triggered.append(rule_result.get("layer", "rule"))
    if semantic_result.get("matched"):
        triggered.append("semantic")

    return {
        "is_sensitive":  is_sensitive,
        "triggered_by":  triggered,
        "rule_result":   rule_result,
        "semantic_result": semantic_result,
    }


# ─── Report Printer ───────────────────────────────────────────────────────────

def print_test_report(results: list[dict]) -> None:
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    tp      = sum(1 for r in results if r["expected"] and r["detected"])
    fp      = sum(1 for r in results if not r["expected"] and r["detected"])
    fn      = sum(1 for r in results if r["expected"] and not r["detected"])
    tn      = sum(1 for r in results if not r["expected"] and not r["detected"])

    print("\n" + "=" * 65)
    print("  DataShield — Company Adapter Test Report")
    print("=" * 65)
    print(f"  Total tests : {total}")
    print(f"  Passed      : {correct}/{total}  "
          f"({'✓ All passed' if correct == total else '✗ Some failed'})")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives  (correctly blocked): {tp}")
    print(f"    True Negatives  (correctly allowed): {tn}")
    print(f"    False Positives (wrongly blocked)  : {fp}")
    print(f"    False Negatives (missed threats)   : {fn}")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"\n  Precision: {precision:.0%}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall:    {recall:.0%}")

    print("\n" + "-" * 65)
    print("  Per-test Results:")
    print("-" * 65)

    for r in results:
        status = "✓ PASS" if r["correct"] else "✗ FAIL"
        flag   = "BLOCK" if r["detected"] else "ALLOW"
        print(f"\n  [{status}] [{r['lang']}] {r['id']}")
        print(f"    Text    : \"{r['text'][:70]}{'...' if len(r['text']) > 70 else ''}\"")
        print(f"    Expected: {'BLOCK' if r['expected'] else 'ALLOW'}  |  "
              f"Got: {flag}  |  Triggered by: {r['triggered_by'] or 'none'}")
        if r.get("term"):
            print(f"    Matched : \"{r['term']}\"")
        if r.get("note"):
            print(f"    Note    : {r['note']}")

    print("\n" + "=" * 65)

    # Recommendations
    if fn > 0:
        print("\n  [TIP] You have False Negatives (missed sensitive prompts).")
        print("    → Add more documents to Step 1 and re-run the pipeline.")
        print("    → Lower the semantic threshold slightly (e.g. 0.78).")
    if fp > 0:
        print("\n  [TIP] You have False Positives (wrongly blocked safe prompts).")
        print("    → In Step 3 admin panel: remove the triggering terms.")
        print("    → Raise the semantic threshold (e.g. 0.88).")
    if correct == total:
        print("\n  All tests passed! Profile is ready to deploy.")
    print()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataShield: Test the company adapter profile"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help="Path to company_profile.json"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  DataShield AI — Step 4: Test Company Adapter")
    print("=" * 65)

    profile = load_json(args.profile)
    if profile is None:
        print(f"\n[ERROR] Could not load profile: {args.profile}")
        print("  Run Steps 1-3 first to generate the profile.")
        sys.exit(1)

    print(f"\n Profile loaded: {args.profile}")
    print(f"  Company:   {profile.get('company_id', 'unknown')}")
    print(f"  Created:   {profile.get('created_at', 'unknown')}")
    meta = profile.get("metadata", {})
    print(f"  Rules:     {meta.get('exact_rules', 0)} exact  |  "
          f"{meta.get('pattern_rules', 0)} patterns  |  "
          f"{meta.get('keyword_count', 0)} keywords")

    # Initialize detection layers
    print("\n Initializing detection layers...")
    rule_engine     = CompanyRuleEngine(profile)
    semantic_layer  = SemanticLayer(profile)

    # Build test cases
    test_cases = BASE_TEST_CASES + generate_profile_specific_tests(profile)
    print(f"\n Running {len(test_cases)} test cases...")

    results = []
    for tc in test_cases:
        text     = tc["text"]
        expected = tc["expected"]

        detection = run_detection(text, rule_engine, semantic_layer)
        detected  = detection["is_sensitive"]
        correct   = (detected == expected)

        # Best matched term for display
        term = (detection["rule_result"].get("term")
                or detection["semantic_result"].get("term", ""))

        results.append({
            "id":          tc["id"],
            "text":        text,
            "lang":        tc.get("lang", "en"),
            "expected":    expected,
            "detected":    detected,
            "correct":     correct,
            "triggered_by": detection["triggered_by"],
            "term":        term,
            "note":        tc.get("note", ""),
        })

    print_test_report(results)
