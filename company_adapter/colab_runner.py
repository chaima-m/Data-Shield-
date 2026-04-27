"""
DataShield AI — Colab Runner
=============================
Runs the entire Layer 2 (Company Adapter) pipeline in one shot.

Run this in Google Colab after cloning the repo:

    !git clone https://github.com/chaima-m/Data-Shield-.git
    %cd Data-Shield-
    %run company_adapter/colab_runner.py

Or step by step with custom docs dir:

    %run company_adapter/colab_runner.py --docs_dir /content/my_company_docs
    %run company_adapter/colab_runner.py --company_id "acme_corp" --threshold 0.80

What this does:
    Step 0 — Install dependencies
    Step 1 — Extract terms from documents in docs_dir
    Step 2 — Build multilingual embeddings + semantic index
    Step 3 — Build company_profile.json
    Step 4 — Run tests and print detection report
    Step 5 — Show profile summary + next steps
"""

import os
import sys
import subprocess
import argparse

# ─── Parse args ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="DataShield Company Adapter — Colab Runner")
parser.add_argument("--docs_dir",   type=str, default="./company_adapter/sample_docs",
                    help="Folder containing company documents")
parser.add_argument("--company_id", type=str, default="my_company",
                    help="Company identifier (e.g. acme_corp)")
parser.add_argument("--threshold",  type=float, default=0.82,
                    help="Semantic similarity threshold (0.0-1.0)")
parser.add_argument("--skip_install", action="store_true",
                    help="Skip pip install (if already installed)")
args = parser.parse_args()

# Make sure we're in the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "company_adapter"))

print("\n" + "=" * 65)
print("  DataShield AI — Layer 2: Company Adapter Pipeline")
print("=" * 65)
print(f"  Docs dir:   {args.docs_dir}")
print(f"  Company ID: {args.company_id}")
print(f"  Threshold:  {args.threshold}")
print("=" * 65)


# ─── Step 0: Install dependencies ────────────────────────────────────────────

if not args.skip_install:
    print("\n[STEP 0] Installing dependencies...")
    req_path = os.path.join(REPO_ROOT, "company_adapter", "requirements.txt")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", req_path],
        check=True
    )
    # Download spaCy model
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "xx_ent_wiki_sm", "--quiet"],
        check=True
    )
    print("  Dependencies installed.")
else:
    print("\n[STEP 0] Skipping install (--skip_install flag set).")


# ─── Step 1: Extract terms ────────────────────────────────────────────────────

print("\n" + "─" * 65)
print("[STEP 1] Extracting terms from company documents...")
print("─" * 65)

from company_adapter import adapter_utils  # noqa: E402 — after sys.path setup
sys.path.insert(0, REPO_ROOT)

import importlib, types

# Dynamically import step modules (avoids top-level import issues)
def import_step(step_file: str):
    spec   = importlib.util.spec_from_file_location(
        f"step_{step_file}", os.path.join(REPO_ROOT, "company_adapter", step_file)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

step1 = import_step("1_extract_terms.py")
terms = step1.extract_from_folder(args.docs_dir)
adapter_utils.save_json(terms, "./adapter_output/extracted_terms.json")

print(f"\n  Entities:      {len(terms.get('entities', []))}")
print(f"  Keywords:      {len(terms.get('keywords', []))}")
print(f"  Bigrams:       {len(terms.get('bigrams', []))}")
print(f"  Code patterns: {len(terms.get('code_patterns', []))}")


# ─── Step 2: Build embeddings ─────────────────────────────────────────────────

print("\n" + "─" * 65)
print("[STEP 2] Building multilingual embeddings + FAISS index...")
print("─" * 65)

step2    = import_step("2_build_embeddings.py")
enriched = step2.build_embeddings(terms)
adapter_utils.save_json(enriched, "./adapter_output/enriched_terms.json")

stats = enriched.get("stats", {})
print(f"\n  Terms before dedup: {stats.get('total_terms_before_dedup', '?')}")
print(f"  Terms after dedup:  {stats.get('total_terms_after_dedup', '?')}")
print(f"  PCA dimensions:     {stats.get('pca_dims', '?')}")


# ─── Step 3: Build profile ────────────────────────────────────────────────────

print("\n" + "─" * 65)
print("[STEP 3] Building company_profile.json...")
print("─" * 65)

step3   = import_step("3_profile_builder.py")
profile = step3.build_profile(
    enriched,
    company_id=args.company_id,
    threshold=args.threshold,
)
adapter_utils.save_json(profile, "./adapter_output/company_profile.json")
lite = step3.build_lite_profile(profile)
adapter_utils.save_json(lite, "./adapter_output/company_profile_lite.json")

m = profile.get("metadata", {})
print(f"\n  Exact match rules:  {m.get('exact_rules', 0)}")
print(f"  Pattern rules:      {m.get('pattern_rules', 0)}")
print(f"  Keywords:           {m.get('keyword_count', 0)}")
print(f"  Semantic vectors:   {m.get('semantic_vectors', 0)}")

full_kb = os.path.getsize("./adapter_output/company_profile.json") / 1024
print(f"\n  Profile size: {full_kb:.1f} KB")


# ─── Step 4: Run tests ────────────────────────────────────────────────────────

print("\n" + "─" * 65)
print("[STEP 4] Running detection tests...")
print("─" * 65)

step4        = import_step("4_test_adapter.py")
rule_engine  = step4.CompanyRuleEngine(profile)
sem_layer    = step4.SemanticLayer(profile)

test_cases   = step4.BASE_TEST_CASES + step4.generate_profile_specific_tests(profile)
results      = []

for tc in test_cases:
    detection = step4.run_detection(tc["text"], rule_engine, sem_layer)
    detected  = detection["is_sensitive"]
    correct   = (detected == tc["expected"])
    term      = (detection["rule_result"].get("term") or
                 detection["semantic_result"].get("term", ""))
    results.append({
        "id":           tc["id"],
        "text":         tc["text"],
        "lang":         tc.get("lang", "en"),
        "expected":     tc["expected"],
        "detected":     detected,
        "correct":      correct,
        "triggered_by": detection["triggered_by"],
        "term":         term,
        "note":         tc.get("note", ""),
    })

step4.print_test_report(results)


# ─── Step 5: Summary ─────────────────────────────────────────────────────────

correct_count = sum(1 for r in results if r["correct"])
total_count   = len(results)

print("\n" + "=" * 65)
print("  Pipeline Complete!")
print("=" * 65)
print(f"\n  Test accuracy: {correct_count}/{total_count} "
      f"({'✓' if correct_count == total_count else '~'} {100*correct_count//total_count}%)")
print("\n  Output files:")
print("    adapter_output/company_profile.json        ← Deploy to extension")
print("    adapter_output/company_profile_lite.json   ← Human-readable review")
print("    adapter_output/extracted_terms.json        ← Raw extraction")
print("    adapter_output/enriched_terms.json         ← After embedding + dedup")

print("\n  Next steps:")
print("  ─────────────────────────────────────────────────")
print("  1. Review company_profile_lite.json")
print("     → Remove any false positive terms manually")
print("  2. Deploy company_profile.json to the extension:")
print("     a. Copy to extension directory")
print("     b. Load via chrome.storage.local.set()")
print("     c. OR deploy via IT Group Policy (see README)")
print("  3. In 4_detection_engine.js, replace CompanyRuleEngine with")
print("     EnhancedCompanyEngine from 5_company_adapter.js")
print("  4. Test in the browser on actual AI tool pages")
print("  ─────────────────────────────────────────────────")
print()
