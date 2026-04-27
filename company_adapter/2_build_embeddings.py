"""
DataShield AI — Step 2: Build Embeddings + Semantic Enrichment
===============================================================
Takes the raw extracted terms and:
  1. Embeds them with a lightweight multilingual sentence-transformer
  2. Uses FAISS to find semantic duplicates → deduplicate & cluster
  3. Expands coverage: finds similar/related terms across the documents
  4. Scores terms by semantic importance (removes noise)
  5. Exports enriched_terms.json ready for the profile builder

Model used: paraphrase-multilingual-MiniLM-L12-v2
  - Size: ~90MB download, ~45MB RAM at inference
  - Dims: 384 → reduced to 64 via PCA for storage efficiency
  - Languages: EN, FR, AR, + 50 others
  - Speed: ~5ms per term on CPU

Input:  ./adapter_output/extracted_terms.json
Output: ./adapter_output/enriched_terms.json

Usage:
    python company_adapter/2_build_embeddings.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

from adapter_utils import load_json, save_json

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_FILE   = "./adapter_output/extracted_terms.json"
OUTPUT_FILE  = "./adapter_output/enriched_terms.json"

# Lightweight multilingual model (90MB, works on CPU, covers EN/FR/AR)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# PCA output dimensions (for storage in profile.json)
# 64 dims gives ~97% variance retention vs full 384 dims
PCA_DIMS = 64

# Cosine similarity threshold for deduplication
# Terms with similarity > this are considered duplicates
DEDUP_THRESHOLD  = 0.92

# Similarity threshold for semantic grouping/clustering
CLUSTER_THRESHOLD = 0.78

# Max terms to embed (keep profile light)
MAX_TERMS_TO_EMBED = 200

# Min confidence score for a term to be kept after scoring
MIN_TERM_SCORE = 0.2


# ─── Embedding Model ──────────────────────────────────────────────────────────

def load_embedding_model():
    """Load the lightweight multilingual sentence transformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        sys.exit(1)

    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    print("  (First run downloads ~90MB — cached for future runs)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def embed_terms(terms: list[str], model) -> np.ndarray:
    """
    Embed a list of text terms.
    Returns a normalized float32 matrix of shape (N, 384).
    """
    print(f"  Embedding {len(terms)} terms...")
    embeddings = model.encode(
        terms,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → cosine = dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ─── PCA Compression ─────────────────────────────────────────────────────────

def compress_embeddings(embeddings: np.ndarray, n_components: int = PCA_DIMS) -> tuple:
    """
    Reduce embedding dimensions with PCA.
    Returns (compressed_embeddings, pca_model).
    Stores PCA params so we can compress query vectors at inference time.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    actual_dims = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=actual_dims, random_state=42)
    compressed = pca.fit_transform(embeddings)
    compressed = normalize(compressed)  # re-normalize after PCA

    variance_ratio = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {embeddings.shape[1]} → {actual_dims} dims, "
          f"variance retained: {variance_ratio:.1%}")

    pca_params = {
        "components":  pca.components_.tolist(),     # shape: (64, 384)
        "mean":        pca.mean_.tolist(),           # shape: (384,)
        "n_components": actual_dims,
        "variance_explained": round(float(variance_ratio), 4),
    }
    return compressed, pca_params


# ─── FAISS Deduplication ──────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    """Build a flat inner-product FAISS index (cosine similarity)."""
    try:
        import faiss
    except ImportError:
        print("[ERROR] faiss-cpu not installed.")
        print("  Run: pip install faiss-cpu")
        sys.exit(1)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine (normalized vectors)
    index.add(embeddings)
    return index


def deduplicate_terms(terms: list[str], embeddings: np.ndarray,
                      threshold: float = DEDUP_THRESHOLD) -> list[int]:
    """
    Remove near-duplicate terms using cosine similarity.
    Returns indices of unique terms to keep.
    """
    import faiss

    index = build_faiss_index(embeddings)
    keep  = []
    removed = set()

    for i in range(len(terms)):
        if i in removed:
            continue
        keep.append(i)

        # Find all terms very similar to terms[i]
        query   = embeddings[i:i+1]
        # k+1 because the term matches itself
        k       = min(len(terms), 20)
        dists, ids = index.search(query, k)

        for dist, j in zip(dists[0], ids[0]):
            if j == i or j in removed:
                continue
            if float(dist) >= threshold:
                removed.add(j)

    print(f"  Dedup: {len(terms)} → {len(keep)} unique terms "
          f"({len(terms) - len(keep)} duplicates removed)")
    return keep


# ─── Term Scoring ─────────────────────────────────────────────────────────────

def score_terms(terms_data: list[dict]) -> list[dict]:
    """
    Score terms by their likely sensitivity importance.
    Combines: entity type weight + frequency + length heuristics.
    """
    TYPE_WEIGHTS = {
        "ORG":        0.8,
        "PRODUCT":    0.9,
        "WORK_OF_ART": 0.95,
        "EVENT":      0.85,
        "FAC":        0.7,
        "LAW":        0.75,
        "GPE":        0.5,
        "KEYWORD":    0.6,
        "BIGRAM":     0.7,
        "CODE":       1.0,
    }

    for t in terms_data:
        entity_type = t.get("type", "KEYWORD")
        type_weight = TYPE_WEIGHTS.get(entity_type, 0.6)

        freq = t.get("count", 1)
        # Log-scale frequency (avoid over-weighting very common generic words)
        freq_score = min(1.0, (1 + np.log(freq)) / 5.0)

        # Longer terms are usually more specific → higher weight
        length = len(t["text"])
        length_score = min(1.0, length / 30.0)

        # Penalize single-word generic terms slightly
        word_count = len(t["text"].split())
        multi_word_bonus = 0.15 if word_count > 1 else 0.0

        score = (type_weight * 0.5 +
                 freq_score  * 0.25 +
                 length_score * 0.1 +
                 multi_word_bonus)

        t["score"] = round(score, 3)

    return terms_data


# ─── Semantic Clustering ──────────────────────────────────────────────────────

def cluster_terms(terms: list[str], embeddings: np.ndarray,
                  threshold: float = CLUSTER_THRESHOLD) -> list[dict]:
    """
    Group semantically related terms into clusters.
    Each cluster has one representative + members.
    Used to enrich keyword lists in the profile.
    """
    import faiss

    index  = build_faiss_index(embeddings)
    visited = set()
    clusters = []

    for i in range(len(terms)):
        if i in visited:
            continue
        visited.add(i)

        query = embeddings[i:i+1]
        k     = min(len(terms), 10)
        dists, ids = index.search(query, k)

        members = [terms[i]]
        for dist, j in zip(dists[0][1:], ids[0][1:]):  # skip self
            if j not in visited and float(dist) >= threshold:
                members.append(terms[j])
                visited.add(j)

        clusters.append({
            "representative": terms[i],
            "members":        members,
            "size":           len(members),
        })

    clusters.sort(key=lambda c: -c["size"])
    return clusters


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def build_embeddings(extracted_terms: dict) -> dict:
    """
    Main pipeline: embed → deduplicate → score → cluster → export.
    """

    # ── Collect all text terms from extraction results ────────────────────────
    all_terms = []

    # Entities (highest priority)
    for e in extracted_terms.get("entities", []):
        all_terms.append({
            "text":  e["text"],
            "type":  e["type"],
            "count": e["count"],
            "source": "ner",
        })

    # Bigrams (project/client names)
    for bg in extracted_terms.get("bigrams", []):
        all_terms.append({
            "text":  bg["text"],
            "type":  "BIGRAM",
            "count": bg["count"],
            "source": "bigram",
        })

    # Keywords (domain vocabulary)
    for kw in extracted_terms.get("keywords", []):
        all_terms.append({
            "text":  kw["text"],
            "type":  "KEYWORD",
            "count": kw["count"],
            "source": "keyword",
        })

    # Code patterns (always kept — don't need embedding)
    code_terms = extracted_terms.get("code_patterns", [])
    for c in code_terms:
        all_terms.append({
            "text":  c["original"],
            "type":  "CODE",
            "count": 1,
            "source": "code_pattern",
        })

    print(f"\n  Total candidate terms: {len(all_terms)}")

    # Limit to top N for embedding (by count)
    all_terms.sort(key=lambda t: -t.get("count", 1))
    terms_to_embed = all_terms[:MAX_TERMS_TO_EMBED]
    term_texts     = [t["text"] for t in terms_to_embed]

    # ── Embed ─────────────────────────────────────────────────────────────────
    model      = load_embedding_model()
    embeddings = embed_terms(term_texts, model)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    print("\n Deduplicating semantically similar terms...")
    unique_indices = deduplicate_terms(term_texts, embeddings)
    unique_terms   = [terms_to_embed[i] for i in unique_indices]
    unique_embeds  = embeddings[unique_indices]
    unique_texts   = [t["text"] for t in unique_terms]

    # ── Score ─────────────────────────────────────────────────────────────────
    print("\n Scoring terms by sensitivity importance...")
    unique_terms = score_terms(unique_terms)
    # Filter low-score noise
    unique_terms = [t for t in unique_terms if t["score"] >= MIN_TERM_SCORE]
    print(f"  After scoring filter: {len(unique_terms)} terms retained")

    # Re-align embeddings after filtering
    keep_mask    = [i for i, t in enumerate(unique_terms) if t["score"] >= MIN_TERM_SCORE]
    unique_embeds = unique_embeds[:len(unique_terms)]

    # ── Compress with PCA ─────────────────────────────────────────────────────
    print("\n Compressing embeddings with PCA...")
    compressed_embeds, pca_params = compress_embeddings(unique_embeds)

    # ── Cluster for coverage insight ──────────────────────────────────────────
    print("\n Clustering related terms...")
    clusters = cluster_terms(unique_texts[:len(unique_terms)], unique_embeds)

    # ── Build output ──────────────────────────────────────────────────────────
    # Store compressed vectors as lists (for JSON serialization)
    terms_with_vectors = []
    for i, term in enumerate(unique_terms):
        entry = dict(term)
        entry["vector"] = compressed_embeds[i].tolist()  # 64-dim float list
        terms_with_vectors.append(entry)

    # Separate code patterns (handled by regex, not embedding)
    code_patterns = [
        {"original": c["original"], "pattern": c["pattern"], "label": "confidential"}
        for c in extracted_terms.get("code_patterns", [])
    ]

    result = {
        "version":       "1.0",
        "embedding_model": EMBEDDING_MODEL,
        "pca": {
            "n_components":        PCA_DIMS,
            "variance_explained":  pca_params["variance_explained"],
            "components":          pca_params["components"],
            "mean":                pca_params["mean"],
        },
        "terms":          terms_with_vectors,
        "clusters":       clusters[:30],   # top 30 clusters for debugging
        "code_patterns":  code_patterns,
        "stats": {
            "total_terms_before_dedup": len(all_terms),
            "total_terms_after_dedup":  len(unique_terms),
            "pca_dims":                 PCA_DIMS,
        },
    }

    return result


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DataShield AI — Step 2: Build Embeddings")
    print("=" * 60)

    extracted = load_json(INPUT_FILE)
    if extracted is None:
        print(f"\n[ERROR] Could not load {INPUT_FILE}")
        print("  Run Step 1 first: python company_adapter/1_extract_terms.py")
        sys.exit(1)

    print(f"\n Loaded extracted terms from: {INPUT_FILE}")
    print(f"   Entities:      {len(extracted.get('entities', []))}")
    print(f"   Keywords:      {len(extracted.get('keywords', []))}")
    print(f"   Bigrams:       {len(extracted.get('bigrams', []))}")
    print(f"   Code patterns: {len(extracted.get('code_patterns', []))}")

    enriched = build_embeddings(extracted)

    save_json(enriched, OUTPUT_FILE)
    print(f"\n Done. Output saved to: {OUTPUT_FILE}")
    print(" Next step: python company_adapter/3_profile_builder.py")
