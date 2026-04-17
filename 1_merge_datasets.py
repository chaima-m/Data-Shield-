"""
DataShield — Fixed Dataset Builder
====================================
KEY FIX: Cap PII at 2000 samples to fix the massive class imbalance
that caused fake F1=1.00 in the original run.

Target: ~12k balanced rows across 6 labels × 3 languages
Runtime: ~5 minutes on Colab
"""

import os, random, re
import pandas as pd
from datasets import load_dataset
from sklearn.utils import resample

random.seed(42)

LABELS   = ["safe", "pii", "financial", "confidential", "health", "credentials"]
LANGS    = ["en", "fr", "ar"]

# ── Hard caps per (label, language) cell ──────────────────────────────────────
PII_CAP      = 2_000   # ← THE FIX: was 203k before
OTHER_CAP    = 700     # target per non-PII cell
SYNTH_TARGET = 600     # synthetic samples per cell if HuggingFace data is short

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Synthetic templates  (unchanged from original, kept compact)
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    "pii": {
        "en": [
            "My name is {name} and I live at {address}.",
            "Contact me at {email} or {phone}.",
            "My date of birth is {dob} and SSN is {ssn}.",
            "Passport number: {passport}, issued to {name}.",
            "Please send the package to {name}, {address}.",
        ],
        "fr": [
            "Je m'appelle {name} et j'habite au {address}.",
            "Contactez-moi à {email} ou au {phone}.",
            "Ma date de naissance est le {dob}.",
            "Mon numéro de passeport est {passport}.",
            "Envoyez le colis à {name}, {address}.",
        ],
        "ar": [
            "اسمي {name} وأسكن في {address}.",
            "تواصل معي على {email} أو {phone}.",
            "تاريخ ميلادي هو {dob}.",
            "رقم جواز سفري هو {passport}.",
            "أرسل الطرد إلى {name}، {address}.",
        ],
    },
    "financial": {
        "en": [
            "My credit card number is {cc} with CVV {cvv}.",
            "Transfer $5,000 to IBAN {iban}.",
            "My bank account is {account} at routing {routing}.",
            "Invoice total: ${amount}, pay to account {account}.",
            "Wire transfer to {iban}, reference {ref}.",
        ],
        "fr": [
            "Mon numéro de carte est {cc}, CVV {cvv}.",
            "Virement de 5000€ vers l'IBAN {iban}.",
            "Mon compte bancaire est {account}.",
            "Facture de {amount}€, payer au compte {account}.",
            "Transfert vers {iban}, référence {ref}.",
        ],
        "ar": [
            "رقم بطاقتي الائتمانية هو {cc} ورمز CVV هو {cvv}.",
            "حوّل 5000 دولار إلى IBAN {iban}.",
            "حسابي البنكي هو {account}.",
            "فاتورة بقيمة {amount} دولار، ادفع إلى الحساب {account}.",
            "تحويل إلى {iban}، المرجع {ref}.",
        ],
    },
    "confidential": {
        "en": [
            "Project {project} is confidential — do not share.",
            "Internal memo: the merger with {company} closes {date}.",
            "Our Q{q} revenue target is ${amount}M.",
            "Client {client} contract value: ${amount}M.",
            "Unreleased product roadmap for {project} attached.",
        ],
        "fr": [
            "Le projet {project} est confidentiel — ne pas divulguer.",
            "Mémo interne : la fusion avec {company} se clôture le {date}.",
            "Notre objectif de chiffre d'affaires Q{q} est {amount}M€.",
            "Contrat client {client} : valeur {amount}M€.",
            "Feuille de route non publiée pour {project} ci-jointe.",
        ],
        "ar": [
            "المشروع {project} سري - لا تشاركه.",
            "مذكرة داخلية: الاندماج مع {company} يُغلق في {date}.",
            "هدف إيرادات الربع Q{q} هو {amount} مليون دولار.",
            "عقد العميل {client}: قيمة {amount} مليون دولار.",
            "خارطة طريق المنتج غير المُصدرة لـ {project} مرفقة.",
        ],
    },
    "health": {
        "en": [
            "Patient {name} diagnosed with {condition}, prescribed {med}.",
            "Lab results for {name}: glucose {value} mg/dL.",
            "Medical record #{record}: {name} admitted for {condition}.",
            "Prescription: {name}, {med} 10mg twice daily.",
            "Blood type O+, HIV negative for patient {name}.",
        ],
        "fr": [
            "Patient {name} diagnostiqué avec {condition}, prescrit {med}.",
            "Résultats de laboratoire pour {name} : glucose {value} mg/dL.",
            "Dossier médical #{record} : {name} admis pour {condition}.",
            "Ordonnance : {name}, {med} 10mg deux fois par jour.",
            "Groupe sanguin O+, VIH négatif pour le patient {name}.",
        ],
        "ar": [
            "المريض {name} مُشخَّص بـ {condition}، وصف له {med}.",
            "نتائج المختبر لـ {name}: جلوكوز {value} ملغ/ديسيلتر.",
            "السجل الطبي #{record}: {name} مُدخل لـ {condition}.",
            "وصفة طبية: {name}، {med} 10 ملغ مرتين يومياً.",
            "فصيلة الدم O+، HIV سلبي للمريض {name}.",
        ],
    },
    "credentials": {
        "en": [
            "Password: {password}, username: {user}@company.com.",
            "API key: sk-prod-{apikey}.",
            "DB connection: postgres://{user}:{password}@{host}/prod.",
            "SSH private key for server {host} stored in ~/.ssh/id_rsa.",
            "AWS access key: AKIA{apikey}, secret: {password}.",
        ],
        "fr": [
            "Mot de passe : {password}, utilisateur : {user}@societe.fr.",
            "Clé API : sk-prod-{apikey}.",
            "Connexion DB : postgres://{user}:{password}@{host}/prod.",
            "Clé SSH pour {host} dans ~/.ssh/id_rsa.",
            "Clé AWS : AKIA{apikey}, secret : {password}.",
        ],
        "ar": [
            "كلمة المرور: {password}، اسم المستخدم: {user}@company.com.",
            "مفتاح API: sk-prod-{apikey}.",
            "اتصال قاعدة البيانات: postgres://{user}:{password}@{host}/prod.",
            "مفتاح SSH للخادم {host} في ~/.ssh/id_rsa.",
            "مفتاح AWS: AKIA{apikey}، السر: {password}.",
        ],
    },
    "safe": {
        "en": [
            "Can you help me write a cover letter for a marketing role?",
            "Summarize this article about climate change.",
            "What is the capital of France?",
            "Translate 'hello world' to Spanish.",
            "Write a Python function to reverse a string.",
        ],
        "fr": [
            "Peux-tu m'aider à rédiger une lettre de motivation pour un poste marketing?",
            "Résume cet article sur le changement climatique.",
            "Quelle est la capitale de la France?",
            "Traduis 'bonjour monde' en espagnol.",
            "Écris une fonction Python pour inverser une chaîne.",
        ],
        "ar": [
            "هل يمكنك مساعدتي في كتابة خطاب تغطية لوظيفة تسويق؟",
            "لخّص هذه المقالة عن تغير المناخ.",
            "ما عاصمة فرنسا؟",
            "ترجم 'مرحبا بالعالم' إلى الإسبانية.",
            "اكتب دالة Python لعكس سلسلة نصية.",
        ],
    },
}

FILLERS = {
    "name": ["Alice Martin", "Jean Dupont", "محمد علي", "Sarah Johnson", "أحمد بن يوسف"],
    "address": ["12 rue de la Paix, Paris", "123 Main St, NY", "شارع الملك فهد، الرياض"],
    "email": ["alice@corp.com", "jean.dupont@example.fr", "ahmed@example.sa"],
    "phone": ["+33 6 12 34 56 78", "+1-555-0199", "+966 50 123 4567"],
    "dob": ["1985-03-22", "22/03/1985", "١٩٨٥/٠٣/٢٢"],
    "ssn": ["123-45-6789", "987-65-4321"],
    "passport": ["AB123456", "FR9876543"],
    "cc": ["4532015112830366", "5425233430109903"],
    "cvv": ["123", "456"],
    "iban": ["FR7630006000011234567890189", "DE89370400440532013000"],
    "account": ["00123456789", "9876543210"],
    "routing": ["021000021", "110000000"],
    "amount": ["5,000", "12,500", "250,000"],
    "ref": ["INV-2024-001", "TRF-98765"],
    "project": ["Phoenix", "Aurora", "Delta-X"],
    "company": ["Acme Corp", "TechGiant Inc", "MegaCorp"],
    "date": ["Q3 2025", "December 31, 2025"],
    "q": ["1", "2", "3", "4"],
    "client": ["ClientAlpha", "BetaCorp", "GammaTech"],
    "condition": ["diabetes", "hypertension", "COVID-19"],
    "med": ["Metformin", "Lisinopril", "Remdesivir"],
    "value": ["120", "95", "210"],
    "record": ["MR-001234", "MR-567890"],
    "password": ["P@ssw0rd!", "Tr0ub4dor&3", "C0rr3ctH0rse"],
    "user": ["admin", "jdupont", "ahmed.admin"],
    "apikey": ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "AbCdEfGhIjKlMnOpQrStUvWxYz123456"],
    "host": ["db.prod.company.com", "api.internal.corp"],
}


def fill(template):
    for k, v in FILLERS.items():
        template = template.replace("{" + k + "}", random.choice(v))
    return template


def generate_synthetic(label, lang, n):
    templates = TEMPLATES[label][lang]
    return [{"text": fill(random.choice(templates)), "label": label, "language": lang}
            for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load HuggingFace datasets
# ─────────────────────────────────────────────────────────────────────────────

def load_ai4privacy(cap=PII_CAP):
    """Load AI4Privacy and cap PII samples — THE KEY FIX."""
    print(f"Loading AI4Privacy (capped at {cap} rows) …")
    rows = []
    try:
        ds = load_dataset("ai4privacy/pii-masking-400k", split="train", streaming=True)
        count = 0
        for ex in ds:
            if count >= cap:
                break
            text = ex.get("source_text") or ex.get("text", "")
            if text and len(text.strip()) > 10:
                rows.append({"text": text.strip(), "label": "pii", "language": "en"})
                count += 1
    except Exception as e:
        print(f"  AI4Privacy load failed: {e} — will use synthetic only for PII")
    print(f"  → {len(rows)} PII rows loaded")
    return rows


def load_multinerd(cap=OTHER_CAP):
    """Load MultiNERD for safe / entity text."""
    print("Loading MultiNERD …")
    rows = []
    try:
        ds = load_dataset("Babelscape/multinerd", split="train", streaming=True)
        counts = {"en": 0, "fr": 0}
        for ex in ds:
            lang = ex.get("lang", "en")
            if lang not in counts or counts[lang] >= cap:
                continue
            tokens = ex.get("tokens", [])
            text = " ".join(tokens)
            if text and len(text.strip()) > 10:
                rows.append({"text": text.strip(), "label": "safe", "language": lang})
                counts[lang] += 1
            if all(v >= cap for v in counts.values()):
                break
    except Exception as e:
        print(f"  MultiNERD load failed: {e}")
    print(f"  → {len(rows)} safe rows from MultiNERD")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build balanced dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    all_rows = []

    # --- PII (capped) ---
    pii_rows = load_ai4privacy(cap=PII_CAP)
    # Add French/Arabic PII via synthetic
    for lang in ["fr", "ar"]:
        pii_rows += generate_synthetic("pii", lang, OTHER_CAP)
    # Cap English PII too
    en_pii = [r for r in pii_rows if r["language"] == "en"][:PII_CAP // 3]
    fr_pii = [r for r in pii_rows if r["language"] == "fr"][:OTHER_CAP]
    ar_pii = [r for r in pii_rows if r["language"] == "ar"][:OTHER_CAP]
    all_rows += en_pii + fr_pii + ar_pii

    # --- Safe (from MultiNERD + synthetic) ---
    safe_rows = load_multinerd(cap=OTHER_CAP)
    for lang in LANGS:
        existing = [r for r in safe_rows if r["language"] == lang]
        needed = max(0, OTHER_CAP - len(existing))
        if needed > 0:
            safe_rows += generate_synthetic("safe", lang, needed)
    all_rows += safe_rows

    # --- Other labels: pure synthetic ---
    for label in ["financial", "confidential", "health", "credentials"]:
        for lang in LANGS:
            all_rows += generate_synthetic(label, lang, OTHER_CAP)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n📊 Final label distribution:")
    print(df["label"].value_counts().to_string())
    print("\n📊 Language distribution:")
    print(df["language"].value_counts().to_string())
    print(f"\nTotal rows: {len(df)}")

    # Verify balance — warn if any class is >5x another
    counts = df["label"].value_counts()
    ratio = counts.max() / counts.min()
    if ratio > 5:
        print(f"\n⚠️  Class imbalance ratio: {ratio:.1f}x — consider adjusting caps")
    else:
        print(f"\n✅ Class balance ratio: {ratio:.1f}x — looks good!")

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = build_dataset()
    out = "data/datashield_dataset.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Dataset saved → {out}")
