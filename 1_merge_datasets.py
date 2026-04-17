"""
DataShield AI — Dataset Merger
================================
Merges multiple public + synthetic datasets into a unified format:
  { "text": "...", "label": "safe|pii|financial|confidential|health|credentials", "language": "en|fr|ar" }

Sources handled:
  1. AI4Privacy (HuggingFace)          → pii
  2. MultiNERD (HuggingFace)           → pii (EN / FR / AR)
  3. CANERCorpus (local CONLL file)    → pii (AR)
  4. FinanceNLP / synthetic financial  → financial
  5. Synthetic credentials             → credentials
  6. Synthetic confidential            → confidential
  7. Synthetic health                  → health
  8. Synthetic safe                    → safe

Usage:
    pip install datasets pandas scikit-learn tqdm
    python 1_merge_datasets.py
"""

import os
import re
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL2ID = {
    "safe": 0,
    "pii": 1,
    "financial": 2,
    "confidential": 3,
    "health": 4,
    "credentials": 5,
}

# Minimum samples per (label, language) cell before oversampling
MIN_PER_CELL = 600

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic normalization — keep original casing for NER models."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove near-duplicate rows using first-64-char hash."""
    df = df.copy()
    df["_hash"] = df["text"].str[:64].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    df = df.drop_duplicates(subset="_hash").drop(columns=["_hash"])
    return df.reset_index(drop=True)


def balance_dataset(df: pd.DataFrame, min_samples: int = MIN_PER_CELL) -> pd.DataFrame:
    """
    Upsample minority (label, language) cells to `min_samples`.
    Does NOT downsample majority cells.
    """
    groups = []
    for (label, lang), group in df.groupby(["label", "language"]):
        if len(group) < min_samples:
            upsampled = resample(
                group,
                replace=True,
                n_samples=min_samples,
                random_state=RANDOM_SEED,
            )
            groups.append(upsampled)
        else:
            groups.append(group)
    return pd.concat(groups, ignore_index=True)


# ─── Source 1: AI4Privacy ──────────────────────────────────────────────────────

def load_ai4privacy() -> pd.DataFrame:
    """
    ai4privacy/pii-masking-200k — columns: source_text, privacy_mask, span_labels
    We use source_text as PII examples and derive language from metadata.
    """
    log.info("Loading AI4Privacy …")
    try:
        ds = load_dataset("ai4privacy/pii-masking-200k", split="train", trust_remote_code=True)
        rows = []
        for ex in tqdm(ds, desc="AI4Privacy"):
            text = clean_text(ex.get("source_text", ""))
            if not text or len(text) < 20:
                continue
            lang = ex.get("language", "en").lower()[:2]
            if lang not in ("en", "fr", "ar"):
                lang = "en"
            rows.append({"text": text, "label": "pii", "language": lang})
        df = pd.DataFrame(rows)
        log.info(f"  AI4Privacy: {len(df)} rows")
        return df
    except Exception as e:
        log.warning(f"  AI4Privacy failed: {e} — skipping")
        return pd.DataFrame(columns=["text", "label", "language"])


# ─── Source 2: MultiNERD ───────────────────────────────────────────────────────

def load_multinerd() -> pd.DataFrame:
    """
    Babelscape/multinerd — token-level NER. We reconstruct sentences and label
    any sentence containing PER/LOC/ORG/MISC entities as 'pii'.
    Languages: EN, FR, AR (and others we discard).
    """
    log.info("Loading MultiNERD …")
    try:
        ds = load_dataset("Babelscape/multinerd", split="train", trust_remote_code=True)
        rows = []
        for ex in tqdm(ds, desc="MultiNERD"):
            lang = ex.get("lang", "en").lower()[:2]
            if lang not in ("en", "fr", "ar"):
                continue
            tokens = ex.get("tokens", [])
            ner_tags = ex.get("ner_tags", [])
            sentence = clean_text(" ".join(tokens))
            if not sentence or len(sentence) < 15:
                continue
            # Any non-O tag → PII sentence
            label = "pii" if any(t != 0 for t in ner_tags) else "safe"
            rows.append({"text": sentence, "label": label, "language": lang})

        df = pd.DataFrame(rows)
        log.info(f"  MultiNERD: {len(df)} rows")
        return df
    except Exception as e:
        log.warning(f"  MultiNERD failed: {e} — skipping")
        return pd.DataFrame(columns=["text", "label", "language"])


# ─── Source 3: CANERCorpus (local CONLL) ──────────────────────────────────────

def load_caner(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    CANERCorpus is a CONLL-format Arabic NER dataset.
    Download from: https://huggingface.co/datasets/CAMeL-Lab/CANERCorpus
    Or pass filepath to a local .conll file.
    """
    log.info("Loading CANERCorpus …")
    rows = []

    # Try HuggingFace first
    try:
        ds = load_dataset("CAMeL-Lab/CANERCorpus", split="train", trust_remote_code=True)
        for ex in tqdm(ds, desc="CANERCorpus"):
            tokens = ex.get("tokens", [])
            tags = ex.get("ner_tags", [])
            sentence = clean_text(" ".join(tokens))
            if not sentence or len(sentence) < 10:
                continue
            label = "pii" if any(t != 0 for t in tags) else "safe"
            rows.append({"text": sentence, "label": label, "language": "ar"})
    except Exception as e:
        log.warning(f"  CANERCorpus HF failed: {e}")

    # Fallback: local file
    if not rows and filepath and Path(filepath).exists():
        tokens, tags = [], []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        sentence = clean_text(" ".join(tokens))
                        label = "pii" if any(t != "O" for t in tags) else "safe"
                        rows.append({"text": sentence, "label": label, "language": "ar"})
                        tokens, tags = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        tags.append(parts[-1])

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["text", "label", "language"])
    log.info(f"  CANERCorpus: {len(df)} rows")
    return df


# ─── Source 4: Synthetic Data Templates ───────────────────────────────────────

# Each entry: (template_string, label, language)
# {SLOT} placeholders are filled with realistic values.

SYNTHETIC_TEMPLATES = {

    # ── PII ──────────────────────────────────────────────────────────────────
    ("en", "pii"): [
        "My name is {FULL_NAME} and my email is {EMAIL}.",
        "Please process the claim for {FULL_NAME}, DOB {DOB}, SSN {SSN}.",
        "Send the invoice to {FULL_NAME} at {ADDRESS}.",
        "The patient {FULL_NAME} can be reached at {PHONE}.",
        "Employee ID {EMP_ID} belongs to {FULL_NAME}, hired on {DATE}.",
        "Passport number {PASSPORT} issued to {FULL_NAME} expires {DATE}.",
        "Driver's license {DL} for {FULL_NAME}, state {STATE}.",
    ],
    ("fr", "pii"): [
        "Mon nom est {FULL_NAME} et mon adresse e-mail est {EMAIL}.",
        "Veuillez traiter le dossier de {FULL_NAME}, né(e) le {DOB}, numéro de sécurité sociale {SSN}.",
        "Envoyez la facture à {FULL_NAME} au {ADDRESS}.",
        "Le patient {FULL_NAME} est joignable au {PHONE}.",
        "Numéro employé {EMP_ID} — {FULL_NAME}, date d'embauche {DATE}.",
    ],
    ("ar", "pii"): [
        "اسمي {FULL_NAME} وبريدي الإلكتروني هو {EMAIL}.",
        "يرجى معالجة ملف {FULL_NAME}، تاريخ الميلاد {DOB}، رقم الهوية {SSN}.",
        "أرسل الفاتورة إلى {FULL_NAME} على العنوان {ADDRESS}.",
        "المريض {FULL_NAME} يمكن التواصل معه على {PHONE}.",
        "رقم الجواز {PASSPORT} صادر لـ {FULL_NAME}.",
    ],

    # ── Financial ────────────────────────────────────────────────────────────
    ("en", "financial"): [
        "Process payment for card number {CARD} expiry {EXPIRY} CVV {CVV}.",
        "Wire {AMOUNT} EUR to IBAN {IBAN}, BIC {BIC}, ref: invoice {INV}.",
        "My bank account is {ACCOUNT} at {BANK}, routing {ROUTING}.",
        "Approve the salary of {AMOUNT} USD for employee {EMP_ID}.",
        "Q3 revenue was {AMOUNT}M, margin {PCT}%, projecting {AMOUNT2}M next quarter.",
        "Authorize transaction {TXN_ID} for {AMOUNT} on account {ACCOUNT}.",
        "Budget for Project {PROJ} is {AMOUNT} USD, current spend {AMOUNT2}.",
    ],
    ("fr", "financial"): [
        "Veuillez traiter le paiement pour la carte {CARD}, expiration {EXPIRY}, CVV {CVV}.",
        "Virement de {AMOUNT} EUR vers l'IBAN {IBAN}, référence facture {INV}.",
        "Mon compte bancaire est {ACCOUNT} à {BANK}, code guichet {ROUTING}.",
        "Approuver le salaire de {AMOUNT} EUR pour l'employé {EMP_ID}.",
        "Le chiffre d'affaires du T3 est de {AMOUNT}M€, marge {PCT}%.",
    ],
    ("ar", "financial"): [
        "يرجى معالجة الدفع للبطاقة رقم {CARD}، تاريخ الانتهاء {EXPIRY}، رمز CVV {CVV}.",
        "تحويل {AMOUNT} يورو إلى IBAN {IBAN}، مرجع الفاتورة {INV}.",
        "حسابي البنكي هو {ACCOUNT} في بنك {BANK}.",
        "الإيرادات الربع الثالث بلغت {AMOUNT} مليون، الهامش {PCT}٪.",
    ],

    # ── Credentials ──────────────────────────────────────────────────────────
    ("en", "credentials"): [
        "The production API key is {API_KEY}, keep it secret.",
        "Database connection: postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        "SSH private key fingerprint {FINGERPRINT}, passphrase {PASSPHRASE}.",
        "AWS credentials: access key {AWS_KEY}, secret {AWS_SECRET}, region {REGION}.",
        "My password is {PASSWORD} — please reset it in the system.",
        "JWT secret for service {SVC} is {JWT_SECRET}, rotate it quarterly.",
        "Slack webhook: {WEBHOOK_URL} — do not share externally.",
        "GitHub PAT: {PAT_TOKEN} with repo and packages scope.",
    ],
    ("fr", "credentials"): [
        "La clé API de production est {API_KEY}, à ne pas divulguer.",
        "Connexion base de données: postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}",
        "Mon mot de passe est {PASSWORD} — veuillez le réinitialiser.",
        "Clé AWS: {AWS_KEY}, secret: {AWS_SECRET}.",
        "Token GitHub: {PAT_TOKEN} — accès lecture/écriture.",
    ],
    ("ar", "credentials"): [
        "مفتاح API للإنتاج هو {API_KEY}، يجب الحفاظ على سريته.",
        "اتصال قاعدة البيانات: postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}",
        "كلمة المرور الخاصة بي هي {PASSWORD}.",
        "مفتاح AWS: {AWS_KEY}، السر: {AWS_SECRET}.",
    ],

    # ── Confidential ─────────────────────────────────────────────────────────
    ("en", "confidential"): [
        "Project {CODENAME} launch is planned for {DATE}, under NDA until then.",
        "Acquisition target: {COMPANY}. Deal value estimated at {AMOUNT}M. Confidential.",
        "Internal memo: restructuring {DEPT} department, {N} positions affected.",
        "Board deck for {QUARTER}: revenue miss of {PCT}% — not for distribution.",
        "M&A due diligence on {COMPANY}: findings attached, eyes only.",
        "Operation {CODENAME}: deploy by {DATE}, notify {FULL_NAME} only.",
        "Headcount plan for FY{YEAR}: reduce {DEPT} by {N}, expand {DEPT2} by {N2}.",
    ],
    ("fr", "confidential"): [
        "Le projet {CODENAME} est prévu pour le {DATE}, sous NDA jusqu'à cette date.",
        "Cible d'acquisition: {COMPANY}. Valeur estimée: {AMOUNT}M. Confidentiel.",
        "Note interne: restructuration du département {DEPT}, {N} postes concernés.",
        "Deck du conseil pour {QUARTER}: manque de revenus de {PCT}% — ne pas diffuser.",
    ],
    ("ar", "confidential"): [
        "مشروع {CODENAME} مقرر إطلاقه في {DATE}، سري حتى الإعلان الرسمي.",
        "هدف الاستحواذ: {COMPANY}، القيمة المقدرة {AMOUNT} مليون. سري للغاية.",
        "مذكرة داخلية: إعادة هيكلة قسم {DEPT}، يتأثر {N} من الموظفين.",
    ],

    # ── Health ────────────────────────────────────────────────────────────────
    ("en", "health"): [
        "Patient {FULL_NAME}, DOB {DOB}, diagnosed with {CONDITION} on {DATE}.",
        "Prescription for {FULL_NAME}: {DRUG} {DOSE}mg twice daily.",
        "Lab results for {FULL_NAME} — HbA1c: {VALUE}, LDL: {VALUE2}.",
        "Mental health note: {FULL_NAME} presents with {CONDITION}, referred to {SPECIALIST}.",
        "Surgical history of {FULL_NAME}: {PROCEDURE} performed {DATE} at {HOSPITAL}.",
        "Insurance claim {CLAIM_ID} for {FULL_NAME}: {PROCEDURE}, cost {AMOUNT}.",
        "Allergy record: {FULL_NAME} is allergic to {ALLERGEN}, epi-pen prescribed.",
    ],
    ("fr", "health"): [
        "Patient {FULL_NAME}, né(e) le {DOB}, diagnostiqué(e) avec {CONDITION} le {DATE}.",
        "Ordonnance pour {FULL_NAME}: {DRUG} {DOSE}mg deux fois par jour.",
        "Résultats d'analyse de {FULL_NAME} — glycémie: {VALUE} mmol/L.",
        "Note psychiatrique: {FULL_NAME} présente {CONDITION}, orienté(e) vers {SPECIALIST}.",
    ],
    ("ar", "health"): [
        "المريض {FULL_NAME}، تاريخ الميلاد {DOB}، تم تشخيصه بـ {CONDITION} بتاريخ {DATE}.",
        "وصفة طبية لـ {FULL_NAME}: {DRUG} {DOSE}ملغ مرتين يومياً.",
        "نتائج التحاليل لـ {FULL_NAME} — سكر الدم: {VALUE}.",
        "سجل الحساسية: {FULL_NAME} لديه حساسية من {ALLERGEN}.",
    ],

    # ── Safe ─────────────────────────────────────────────────────────────────
    ("en", "safe"): [
        "Can you summarize the key points of this article for me?",
        "Write a Python function that sorts a list of integers.",
        "What are the best practices for remote team management?",
        "Explain the difference between TCP and UDP protocols.",
        "Draft an email thanking a colleague for their help on a project.",
        "How do I set up a virtual environment in Python?",
        "What is the capital of Australia?",
        "Give me 5 ideas for a team-building activity.",
        "Help me write a cover letter for a software engineer role.",
        "Translate 'good morning' into Spanish.",
    ],
    ("fr", "safe"): [
        "Pouvez-vous résumer les points clés de cet article?",
        "Écris une fonction Python qui trie une liste d'entiers.",
        "Quelles sont les meilleures pratiques pour la gestion d'équipes à distance?",
        "Explique la différence entre TCP et UDP.",
        "Rédige un email de remerciement à un collègue.",
        "Comment configurer un environnement virtuel Python?",
        "Quelle est la capitale de l'Australie?",
        "Donne-moi 5 idées d'activité d'équipe.",
    ],
    ("ar", "safe"): [
        "هل يمكنك تلخيص النقاط الرئيسية لهذا المقال؟",
        "اكتب دالة Python لترتيب قائمة من الأعداد الصحيحة.",
        "ما هي أفضل الممارسات لإدارة الفرق عن بُعد؟",
        "اشرح الفرق بين TCP و UDP.",
        "اكتب بريداً إلكترونياً لشكر زميل على مساعدته.",
        "كيف أُعد بيئة Python الافتراضية؟",
        "ما هي عاصمة أستراليا؟",
    ],
}

# ─── Slot fillers ──────────────────────────────────────────────────────────────

SLOT_VALUES = {
    "FULL_NAME":    ["Alice Martin", "Mohammed Al-Rashid", "Sophie Dubois",
                     "Youssef Benali", "James Chen", "Fatima Zahra", "Carlos Ruiz"],
    "EMAIL":        ["alice.martin@corp.com", "m.alrashid@example.org",
                     "s.dubois@entreprise.fr", "j.chen@startup.io"],
    "DOB":          ["15/03/1985", "2001-07-22", "April 5, 1993"],
    "SSN":          ["123-45-6789", "987-65-4321", "456-78-9012"],
    "ADDRESS":      ["12 Rue Lafayette, Paris 75009",
                     "42 Oak Street, Boston MA 02101",
                     "شارع الملك فهد، الرياض 12345"],
    "PHONE":        ["+1-202-555-0147", "+33 6 12 34 56 78", "+213 555 123 456"],
    "EMP_ID":       ["EMP-00471", "HR-2024-892", "STAFF-115"],
    "DATE":         ["2024-12-01", "March 15, 2025", "01/06/2025"],
    "STATE":        ["California", "New York", "Texas"],
    "PASSPORT":     ["AB1234567", "P9876543", "FR-2819034"],
    "DL":           ["DL-4892301", "CA-DL-77821"],
    "CARD":         ["4111 1111 1111 1111", "5500 0000 0000 0004", "3714 496353 98431"],
    "EXPIRY":       ["09/27", "12/26", "03/28"],
    "CVV":          ["737", "123", "456"],
    "AMOUNT":       ["50,000", "1.2", "250", "3.7", "12,500"],
    "AMOUNT2":      ["55,000", "1.5", "300"],
    "IBAN":         ["DE89370400440532013000", "FR7630006000011234567890189"],
    "BIC":          ["DEUTDEDB", "BNPAFRPP"],
    "ACCOUNT":      ["0012345678", "FR76-3000"],
    "BANK":         ["BNP Paribas", "Deutsche Bank", "Bank of America"],
    "ROUTING":      ["021000021", "026009593"],
    "INV":          ["INV-2024-441", "FACT-0089"],
    "TXN_ID":       ["TXN-8829401", "REF-2024-7712"],
    "PCT":          ["12.5", "8.3", "22"],
    "PROJ":         ["Alpha", "Omega", "Phoenix"],
    "API_KEY":      ["sk-prod-xK9mN2pL8qR5vY3w", "AIzaSyD-9tSrk8aL7_xZ4pM",
                     "AKIAIOSFODNN7EXAMPLE"],
    "DB_USER":      ["prod_admin", "app_svc", "readonly_user"],
    "DB_PASS":      ["S3cur3P@ss!", "Xk9#mN2pL", "db_secret_2024"],
    "DB_HOST":      ["db.internal.corp.com", "10.0.1.45", "prod-pg.cluster.local"],
    "DB_PORT":      ["5432", "3306", "27017"],
    "DB_NAME":      ["production_db", "customer_data", "analytics"],
    "FINGERPRINT":  ["SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8"],
    "PASSPHRASE":   ["MyStr0ngPassphrase!"],
    "AWS_KEY":      ["AKIAIOSFODNN7EXAMPLE", "AKIAI44QH8DHBEXAMPLE"],
    "AWS_SECRET":   ["wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"],
    "REGION":       ["us-east-1", "eu-west-1"],
    "SVC":          ["auth-service", "payment-api"],
    "JWT_SECRET":   ["hs256-secret-2024-prod-xk9m"],
    "WEBHOOK_URL":  ["https://hooks.slack.com/services/T00/B00/xxxx"],
    "PAT_TOKEN":    ["ghp_16C7e42F292c6912E7710c838347Dd6F68Rf6"],
    "PASSWORD":     ["P@ssw0rd!2024", "Xk9#mN2pL!", "Tr0ub4dor&3"],
    "CODENAME":     ["Project Helios", "Operation Sunrise", "Initiative Falcon",
                     "مشروع الفجر", "Projet Aurore"],
    "COMPANY":      ["TechCorp Inc.", "DataSystems Ltd.", "Innovate SA"],
    "DEPT":         ["Engineering", "Sales", "Operations", "R&D"],
    "N":            ["12", "5", "30"],
    "N2":           ["8", "15", "20"],
    "QUARTER":      ["Q3 2024", "Q4 2025", "H1 2025"],
    "YEAR":         ["2025", "2026"],
    "DEPT2":        ["Cloud Infrastructure", "AI/ML", "Customer Success"],
    "CONDITION":    ["Type 2 Diabetes", "Major Depressive Disorder", "Hypertension",
                     "داء السكري", "Diabète de type 2"],
    "DRUG":         ["Metformin", "Sertraline", "Lisinopril", "ميتفورمين"],
    "DOSE":         ["500", "100", "10"],
    "VALUE":        ["7.8", "5.2", "145"],
    "VALUE2":       ["3.4", "2.1"],
    "SPECIALIST":   ["Dr. Chen (Psychiatry)", "Cardiology Dept."],
    "PROCEDURE":    ["Appendectomy", "Coronary angioplasty", "Knee arthroscopy"],
    "HOSPITAL":     ["City General Hospital", "CHU Bordeaux"],
    "CLAIM_ID":     ["CLM-2024-88901", "INS-7712"],
    "ALLERGEN":     ["Penicillin", "Latex", "Shellfish", "البنسلين"],
}


def fill_template(template: str) -> str:
    """Replace all {SLOT} placeholders with random realistic values."""
    def replacer(match):
        slot = match.group(1)
        choices = SLOT_VALUES.get(slot, [f"[{slot}]"])
        return random.choice(choices)
    return re.sub(r"\{(\w+)\}", replacer, template)


def generate_synthetic(n_per_template: int = 80) -> pd.DataFrame:
    """Generate synthetic samples from all templates."""
    log.info("Generating synthetic samples …")
    rows = []
    for (lang, label), templates in SYNTHETIC_TEMPLATES.items():
        for template in templates:
            for _ in range(n_per_template):
                text = fill_template(template)
                rows.append({"text": clean_text(text), "label": label, "language": lang})
    df = pd.DataFrame(rows)
    log.info(f"  Synthetic: {len(df)} rows")
    return df


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def merge_all(caner_filepath: Optional[str] = None, synthetic_multiplier: int = 80) -> pd.DataFrame:
    """
    Load all sources, deduplicate, balance, and return unified DataFrame.

    Args:
        caner_filepath:      Path to local CANER .conll file (optional, HF tried first).
        synthetic_multiplier: How many samples to generate per template line.
    """
    frames = [
        load_ai4privacy(),
        load_multinerd(),
        load_caner(caner_filepath),
        generate_synthetic(n_per_template=synthetic_multiplier),
    ]

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    log.info(f"Combined before dedup: {len(combined)} rows")

    combined = deduplicate(combined)
    log.info(f"After dedup: {len(combined)} rows")

    # Validate labels and languages
    combined = combined[combined["label"].isin(LABEL2ID.keys())]
    combined = combined[combined["language"].isin(("en", "fr", "ar"))]
    combined = combined[combined["text"].str.len() >= 10]

    log.info("Distribution before balancing:")
    print(combined.groupby(["label", "language"]).size().to_string())

    combined = balance_dataset(combined, min_samples=MIN_PER_CELL)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    log.info(f"\nFinal dataset size: {len(combined)} rows")
    log.info("Final distribution:")
    print(combined.groupby(["label", "language"]).size().to_string())

    return combined


def save_dataset(df: pd.DataFrame):
    """Save in multiple formats for convenience."""
    # CSV (primary)
    csv_path = OUTPUT_DIR / "datashield_dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved CSV → {csv_path}")

    # JSONL (for streaming loaders)
    jsonl_path = OUTPUT_DIR / "datashield_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"text": row["text"], "label": row["label"],
                                "language": row["language"]}, ensure_ascii=False) + "\n")
    log.info(f"Saved JSONL → {jsonl_path}")

    # Per-language splits
    for lang in ("en", "fr", "ar"):
        subset = df[df["language"] == lang]
        subset.to_csv(OUTPUT_DIR / f"datashield_{lang}.csv", index=False, encoding="utf-8-sig")

    log.info("All files saved.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataShield Dataset Merger")
    parser.add_argument("--caner-file", default=None, help="Path to CANERCorpus .conll file")
    parser.add_argument("--synth-mult", type=int, default=80,
                        help="Synthetic samples per template (default: 80)")
    args = parser.parse_args()

    df = merge_all(caner_filepath=args.caner_file, synthetic_multiplier=args.synth_mult)
    save_dataset(df)
    print("\n✅ Dataset ready in ./data/")
