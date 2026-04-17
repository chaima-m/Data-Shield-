"""
DataShield AI — Fast Fine-Tuning with SetFit
=============================================
WHAT CHANGED vs previous version:
  - Uses SetFit (sentence-transformers + few-shot) instead of full fine-tuning
  - Training time: ~8-12 minutes on Colab GPU (vs 1+ hour before)
  - Perfectly balanced dataset: 300 samples per label per language = 5400 total
  - Model: paraphrase-multilingual-MiniLM-L12-v2 (~120MB, fast CPU inference)
  - Auto-saves to Google Drive + exports ONNX in same script
  - No optimum needed for ONNX export (uses torch.onnx directly)

COLAB SETUP (run these first):
    !pip install setfit sentence-transformers onnx onnxruntime scikit-learn pandas -q
    from google.colab import drive
    drive.mount('/content/drive')

THEN RUN:
    !python datashield_fast.py

OUTPUT:
    /content/drive/MyDrive/datashield/model_final/     ← SetFit model
    /content/drive/MyDrive/datashield/model_onnx/      ← ONNX fp32
    /content/drive/MyDrive/datashield/model_quantized/ ← ONNX INT8 ← use in browser
"""

import os
import json
import random
import re
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(H:%M:%S} | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Fix logging format ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

# ─── Config ────────────────────────────────────────────────────────────────────

SAVE_DIR   = Path("/content/drive/MyDrive/datashield")   # change if needed
LOCAL_DIR  = Path("/content/datashield_output")
SAMPLES_PER_CLASS_PER_LANG = 300   # 300 × 6 labels × 3 langs = 5400 total
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

for d in (SAVE_DIR, LOCAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Labels ───────────────────────────────────────────────────────────────────

LABELS = ["safe", "pii", "financial", "confidential", "health", "credentials"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# ─── Slot values for synthetic generation ─────────────────────────────────────

SLOTS = {
    "NAME":       ["Alice Martin", "Mohammed Al-Rashid", "Sophie Dubois", "Youssef Benali",
                   "James Chen", "Fatima Zahra", "Carlos Ruiz", "Aisha Ndiaye"],
    "EMAIL":      ["alice.martin@corp.com", "m.rashid@example.org", "j.chen@startup.io",
                   "s.dubois@entreprise.fr", "y.benali@company.dz"],
    "PHONE":      ["+1-202-555-0147", "+33 6 12 34 56 78", "+213 555 123 456",
                   "+44 20 7946 0958", "07 98 76 54 32"],
    "SSN":        ["123-45-6789", "987-65-4321", "456-78-9012", "234-56-7890"],
    "ADDRESS":    ["12 Rue Lafayette, Paris 75009", "42 Oak Street, Boston MA 02101",
                   "شارع الملك فهد، الرياض 12345", "10 Downing Street, London"],
    "DOB":        ["15/03/1985", "2001-07-22", "April 5, 1993", "22 janvier 1990"],
    "PASSPORT":   ["AB1234567", "P9876543", "FR2819034"],
    "CARD":       ["4111 1111 1111 1111", "5500 0000 0000 0004", "3714 496353 98431"],
    "EXPIRY":     ["09/27", "12/26", "03/28"],
    "CVV":        ["737", "123", "456"],
    "IBAN":       ["DE89370400440532013000", "FR7630006000011234567890189", "DZ5900200100130000000054"],
    "AMOUNT":     ["50,000", "1.2M", "250,000", "3.7", "12,500"],
    "BANK":       ["BNP Paribas", "Deutsche Bank", "Bank of America", "CIB Algeria"],
    "API_KEY":    ["sk-prod-xK9mN2pL8qR5vY3w", "AIzaSyD-9tSrk8aL7_xZ4pM",
                   "AKIAIOSFODNN7EXAMPLE", "ghp_16C7e42F292c6912E7710c838347Dd6F"],
    "DB_CONN":    ["postgresql://admin:S3cur3P@ss@db.internal.corp.com:5432/prod",
                   "mysql://root:Xk9mN2pL@10.0.1.45:3306/customers",
                   "mongodb://app_svc:secret@prod-mongo:27017/analytics"],
    "PASSWORD":   ["P@ssw0rd!2024", "Xk9#mN2pL!", "Tr0ub4dor&3", "Admin@2024"],
    "CODENAME":   ["Project Helios", "Operation Sunrise", "Initiative Falcon",
                   "مشروع الفجر", "Projet Aurore", "Project Phoenix"],
    "COMPANY":    ["TechCorp Inc.", "DataSystems Ltd.", "Innovate SA", "NovaTech"],
    "DEPT":       ["Engineering", "Sales", "Operations", "R&D", "Finance"],
    "DRUG":       ["Metformin", "Sertraline", "Lisinopril", "ميتفورمين", "Amoxicilline"],
    "DOSE":       ["500mg", "100mg", "10mg", "250mg"],
    "CONDITION":  ["Type 2 Diabetes", "Hypertension", "Major Depressive Disorder",
                   "داء السكري", "Diabète de type 2", "hypertension artérielle"],
    "HOSPITAL":   ["City General Hospital", "CHU Bordeaux", "مستشفى الملك فيصل"],
    "CLAIM_ID":   ["CLM-2024-88901", "INS-7712", "CLAIM-2025-001"],
    "ALLERGEN":   ["Penicillin", "Latex", "Shellfish", "البنسلين", "pénicilline"],
    "QUARTER":    ["Q3 2024", "Q4 2025", "T3 2024"],
    "DATE":       ["2024-12-01", "March 15, 2025", "01/06/2025", "15 mars 2025"],
    "ROUTE":      ["52.378", "10.0.1.45", "192.168.1.100"],
    "SVC":        ["auth-service", "payment-api", "user-management"],
}

def fill(template: str) -> str:
    def replace(m):
        key = m.group(1)
        return random.choice(SLOTS.get(key, [f"[{key}]"]))
    return re.sub(r"\{(\w+)\}", replace, template)

# ─── Template bank ────────────────────────────────────────────────────────────
# Each tuple: (template, label, language)
# We need variety — multiple phrasings per category

TEMPLATES = [

    # ════════════════════════════════════════════════════════ SAFE ═══
    # English safe
    ("Can you summarize this document for me?", "safe", "en"),
    ("What are the best practices for remote team management?", "safe", "en"),
    ("Explain the difference between TCP and UDP.", "safe", "en"),
    ("Write a Python function that sorts a list of integers.", "safe", "en"),
    ("What is the capital of Australia?", "safe", "en"),
    ("How do I set up a virtual environment in Python?", "safe", "en"),
    ("Draft a professional email thanking a client for their business.", "safe", "en"),
    ("Give me 5 ideas for a team-building activity.", "safe", "en"),
    ("What is machine learning and how does it work?", "safe", "en"),
    ("Help me write a cover letter for a software engineer role.", "safe", "en"),
    ("What's the difference between REST and GraphQL?", "safe", "en"),
    ("How do I optimize a slow SQL query?", "safe", "en"),
    ("Translate 'good morning' into Spanish.", "safe", "en"),
    ("What are the main causes of climate change?", "safe", "en"),
    ("Write a function to check if a number is prime.", "safe", "en"),
    ("How does HTTPS work?", "safe", "en"),
    ("What is the agile methodology?", "safe", "en"),
    ("Explain recursion with a simple example.", "safe", "en"),
    ("What are design patterns in software engineering?", "safe", "en"),
    ("How do I center a div in CSS?", "safe", "en"),
    # French safe
    ("Pouvez-vous résumer les points clés de cet article?", "safe", "fr"),
    ("Écris une fonction Python qui trie une liste d'entiers.", "safe", "fr"),
    ("Quelles sont les meilleures pratiques pour la gestion d'équipes à distance?", "safe", "fr"),
    ("Explique la différence entre TCP et UDP.", "safe", "fr"),
    ("Comment configurer un environnement virtuel Python?", "safe", "fr"),
    ("Quelle est la capitale de l'Australie?", "safe", "fr"),
    ("Rédige un email de remerciement à un collègue.", "safe", "fr"),
    ("Donne-moi 5 idées d'activité d'équipe.", "safe", "fr"),
    ("Comment fonctionne le protocole HTTPS?", "safe", "fr"),
    ("Qu'est-ce que l'intelligence artificielle?", "safe", "fr"),
    ("Comment optimiser une requête SQL lente?", "safe", "fr"),
    ("Explique la récursivité avec un exemple simple.", "safe", "fr"),
    ("Quels sont les principaux paradigmes de programmation?", "safe", "fr"),
    ("Comment centrer un élément div en CSS?", "safe", "fr"),
    ("Qu'est-ce que la méthode agile?", "safe", "fr"),
    # Arabic safe
    ("هل يمكنك تلخيص النقاط الرئيسية لهذا المقال؟", "safe", "ar"),
    ("اكتب دالة Python لترتيب قائمة من الأعداد الصحيحة.", "safe", "ar"),
    ("ما هي أفضل الممارسات لإدارة الفرق عن بُعد؟", "safe", "ar"),
    ("اشرح الفرق بين TCP و UDP.", "safe", "ar"),
    ("اكتب بريداً إلكترونياً لشكر زميل على مساعدته.", "safe", "ar"),
    ("كيف أُعد بيئة Python الافتراضية؟", "safe", "ar"),
    ("ما هي عاصمة أستراليا؟", "safe", "ar"),
    ("اشرح مفهوم التعلم الآلي بطريقة بسيطة.", "safe", "ar"),
    ("ما الفرق بين REST و GraphQL؟", "safe", "ar"),
    ("كيف يعمل بروتوكول HTTPS؟", "safe", "ar"),
    ("ما هو نمط التصميم في هندسة البرمجيات؟", "safe", "ar"),
    ("كيف أحسّن استعلام SQL بطيء؟", "safe", "ar"),

    # ════════════════════════════════════════════════════════ PII ════
    # English PII
    ("My name is {NAME} and my email is {EMAIL}.", "pii", "en"),
    ("Please update the record for {NAME}, SSN {SSN}, DOB {DOB}.", "pii", "en"),
    ("Send the invoice to {NAME} at {ADDRESS}, phone {PHONE}.", "pii", "en"),
    ("Employee {NAME} can be reached at {EMAIL} or {PHONE}.", "pii", "en"),
    ("Passport number {PASSPORT} belongs to {NAME}, DOB {DOB}.", "pii", "en"),
    ("Please verify the identity of {NAME}, born {DOB}, living at {ADDRESS}.", "pii", "en"),
    ("The customer {NAME} reported an issue, their contact is {PHONE}.", "pii", "en"),
    ("Update the HR file for {NAME}: new address is {ADDRESS}.", "pii", "en"),
    ("Notify {NAME} at {EMAIL} about the policy renewal.", "pii", "en"),
    ("Candidate {NAME}, DOB {DOB}, applied for the position. Contact: {PHONE}.", "pii", "en"),
    ("My social security number is {SSN} and I live at {ADDRESS}.", "pii", "en"),
    ("Please find attached the personal details of {NAME}: email {EMAIL}, phone {PHONE}.", "pii", "en"),
    # French PII
    ("Mon nom est {NAME} et mon adresse e-mail est {EMAIL}.", "pii", "fr"),
    ("Veuillez mettre à jour le dossier de {NAME}, né(e) le {DOB}, à l'adresse {ADDRESS}.", "pii", "fr"),
    ("Envoyez la facture à {NAME} au {ADDRESS}, téléphone {PHONE}.", "pii", "fr"),
    ("L'employé(e) {NAME} peut être contacté(e) à {EMAIL} ou {PHONE}.", "pii", "fr"),
    ("Numéro de passeport {PASSPORT} appartenant à {NAME}, date de naissance {DOB}.", "pii", "fr"),
    ("Veuillez vérifier l'identité de {NAME}, né(e) le {DOB}, domicilié(e) au {ADDRESS}.", "pii", "fr"),
    ("Le client {NAME} a signalé un problème, son contact est {PHONE}.", "pii", "fr"),
    ("Mettez à jour le dossier RH de {NAME}: nouvelle adresse {ADDRESS}.", "pii", "fr"),
    ("Notifiez {NAME} à {EMAIL} concernant le renouvellement de la police.", "pii", "fr"),
    ("Mon numéro de sécurité sociale est {SSN} et j'habite au {ADDRESS}.", "pii", "fr"),
    # Arabic PII
    ("اسمي {NAME} وبريدي الإلكتروني هو {EMAIL}.", "pii", "ar"),
    ("يرجى تحديث سجل {NAME}، تاريخ الميلاد {DOB}، العنوان: {ADDRESS}.", "pii", "ar"),
    ("أرسل الفاتورة إلى {NAME} على العنوان {ADDRESS}، الهاتف {PHONE}.", "pii", "ar"),
    ("يمكن التواصل مع الموظف {NAME} عبر {EMAIL} أو {PHONE}.", "pii", "ar"),
    ("رقم جواز السفر {PASSPORT} يعود لـ {NAME}، تاريخ الميلاد {DOB}.", "pii", "ar"),
    ("رقم الهوية الوطنية: {SSN}، الاسم: {NAME}، العنوان: {ADDRESS}.", "pii", "ar"),
    ("تفاصيل العميل: {NAME}، الهاتف {PHONE}، البريد {EMAIL}.", "pii", "ar"),

    # ════════════════════════════════════════════════ FINANCIAL ══════
    # English financial
    ("Process payment for card {CARD}, expiry {EXPIRY}, CVV {CVV}.", "financial", "en"),
    ("Wire {AMOUNT} EUR to IBAN {IBAN}, reference: invoice 2024-441.", "financial", "en"),
    ("My bank account {IBAN} at {BANK} needs to be updated.", "financial", "en"),
    ("Approve the salary of {AMOUNT} USD for employee ID EMP-0047.", "financial", "en"),
    ("Q3 revenue was {AMOUNT}, projecting {AMOUNT} next quarter.", "financial", "en"),
    ("Authorize transaction TXN-8829401 for {AMOUNT} on account {IBAN}.", "financial", "en"),
    ("The budget for {CODENAME} is {AMOUNT}, current spend {AMOUNT}.", "financial", "en"),
    ("My credit card ending in 1111 has a limit of {AMOUNT}.", "financial", "en"),
    ("Transfer {AMOUNT} from account 00123456 to {IBAN} today.", "financial", "en"),
    ("Invoice INV-2024-441 for {AMOUNT} is overdue, please process.", "financial", "en"),
    ("The acquisition deal for {COMPANY} is valued at {AMOUNT}.", "financial", "en"),
    ("Employee bonus for {DEPT}: {AMOUNT} per person this quarter.", "financial", "en"),
    # French financial
    ("Veuillez traiter le paiement pour la carte {CARD}, expiration {EXPIRY}, CVV {CVV}.", "financial", "fr"),
    ("Virement de {AMOUNT} EUR vers l'IBAN {IBAN}, référence facture 2024-441.", "financial", "fr"),
    ("Mon compte bancaire {IBAN} à {BANK} doit être mis à jour.", "financial", "fr"),
    ("Approuver le salaire de {AMOUNT} EUR pour l'employé EMP-0047.", "financial", "fr"),
    ("Le chiffre d'affaires T3 est de {AMOUNT}, projection T4: {AMOUNT}.", "financial", "fr"),
    ("Autoriser la transaction TXN-8829401 de {AMOUNT} sur le compte {IBAN}.", "financial", "fr"),
    ("Budget du projet {CODENAME}: {AMOUNT}, dépensé à ce jour: {AMOUNT}.", "financial", "fr"),
    ("Transférer {AMOUNT} du compte 00123456 vers {IBAN} aujourd'hui.", "financial", "fr"),
    ("La facture INV-2024-441 de {AMOUNT} est en retard, veuillez traiter.", "financial", "fr"),
    # Arabic financial
    ("يرجى معالجة الدفع للبطاقة رقم {CARD}، تاريخ الانتهاء {EXPIRY}، CVV {CVV}.", "financial", "ar"),
    ("تحويل {AMOUNT} يورو إلى IBAN {IBAN}، مرجع الفاتورة 2024-441.", "financial", "ar"),
    ("حسابي البنكي {IBAN} في بنك {BANK} يحتاج تحديثاً.", "financial", "ar"),
    ("الإيرادات للربع الثالث بلغت {AMOUNT}، التوقعات للربع الرابع: {AMOUNT}.", "financial", "ar"),
    ("تفويض معاملة TXN-8829401 بمبلغ {AMOUNT} على الحساب {IBAN}.", "financial", "ar"),
    ("فاتورة INV-2024-441 بمبلغ {AMOUNT} متأخرة، يرجى المعالجة.", "financial", "ar"),

    # ══════════════════════════════════════════════ CONFIDENTIAL ═════
    # English confidential
    ("{CODENAME} launch is planned for {DATE}, under NDA until then.", "confidential", "en"),
    ("Acquisition target: {COMPANY}. Deal value {AMOUNT}. Eyes only.", "confidential", "en"),
    ("Internal memo: restructuring {DEPT} dept, 12 positions affected.", "confidential", "en"),
    ("Board deck for {QUARTER}: revenue miss of 12% — not for distribution.", "confidential", "en"),
    ("M&A due diligence on {COMPANY}: findings confidential.", "confidential", "en"),
    ("Operation {CODENAME}: deploy by {DATE}, notify leadership only.", "confidential", "en"),
    ("Headcount plan FY2025: reduce {DEPT} by 15, expand Cloud by 20.", "confidential", "en"),
    ("Pricing strategy for {COMPANY} partnership: {AMOUNT} per seat.", "confidential", "en"),
    ("Pre-announcement: {COMPANY} merger expected to close {DATE}. NDA applies.", "confidential", "en"),
    ("Executive compensation package for Q4: {AMOUNT} total, confidential.", "confidential", "en"),
    ("{CODENAME} product roadmap is embargoed until official launch.", "confidential", "en"),
    ("Internal audit findings for {DEPT}: do not share outside leadership.", "confidential", "en"),
    # French confidential
    ("Le projet {CODENAME} est prévu pour le {DATE}, sous NDA jusqu'à cette date.", "confidential", "fr"),
    ("Cible d'acquisition: {COMPANY}. Valeur: {AMOUNT}. Confidentiel.", "confidential", "fr"),
    ("Note interne: restructuration du département {DEPT}, 12 postes concernés.", "confidential", "fr"),
    ("Deck du conseil pour {QUARTER}: manque de revenus de 12% — ne pas diffuser.", "confidential", "fr"),
    ("Diligence raisonnable sur {COMPANY}: résultats confidentiels.", "confidential", "fr"),
    ("Plan d'effectifs FY2025: réduire {DEPT} de 15, étendre Cloud de 20.", "confidential", "fr"),
    ("Stratégie tarifaire pour le partenariat {COMPANY}: {AMOUNT} par siège.", "confidential", "fr"),
    ("Annonce anticipée: fusion {COMPANY} attendue pour le {DATE}. NDA applicable.", "confidential", "fr"),
    ("Résultats de l'audit interne pour {DEPT}: ne pas partager hors direction.", "confidential", "fr"),
    # Arabic confidential
    ("مشروع {CODENAME} مقرر إطلاقه في {DATE}، سري حتى الإعلان الرسمي.", "confidential", "ar"),
    ("هدف الاستحواذ: {COMPANY}، القيمة المقدرة {AMOUNT}. للعيون فقط.", "confidential", "ar"),
    ("مذكرة داخلية: إعادة هيكلة قسم {DEPT}، 12 وظيفة متأثرة.", "confidential", "ar"),
    ("خطة التوظيف للسنة المالية 2025: تخفيض {DEPT} بمقدار 15.", "confidential", "ar"),
    ("نتائج التدقيق الداخلي لـ {DEPT}: لا تشاركها خارج القيادة.", "confidential", "ar"),
    ("استراتيجية التسعير لشراكة {COMPANY}: {AMOUNT} لكل مقعد. سري.", "confidential", "ar"),

    # ═══════════════════════════════════════════════════ HEALTH ══════
    # English health
    ("Patient {NAME}, DOB {DOB}, diagnosed with {CONDITION} on {DATE}.", "health", "en"),
    ("Prescription for {NAME}: {DRUG} {DOSE} twice daily.", "health", "en"),
    ("Lab results for {NAME}: HbA1c 7.8, LDL 3.4. Follow up required.", "health", "en"),
    ("Mental health note: {NAME} presents with {CONDITION}, referred to psychiatry.", "health", "en"),
    ("Surgical history of {NAME}: appendectomy performed {DATE} at {HOSPITAL}.", "health", "en"),
    ("Insurance claim {CLAIM_ID} for {NAME}: knee surgery, cost {AMOUNT}.", "health", "en"),
    ("Allergy record: {NAME} is allergic to {ALLERGEN}, epi-pen prescribed.", "health", "en"),
    ("ICU admission: {NAME}, {CONDITION}, admitted {DATE} to {HOSPITAL}.", "health", "en"),
    ("Vaccination record for {NAME}: COVID booster administered {DATE}.", "health", "en"),
    ("Therapy session notes for {NAME}: treatment for {CONDITION} ongoing.", "health", "en"),
    ("Discharge summary for {NAME}: post-op {CONDITION}, prescribed {DRUG} {DOSE}.", "health", "en"),
    ("Medical history: {NAME}, DOB {DOB} — chronic {CONDITION} since 2018.", "health", "en"),
    # French health
    ("Patient {NAME}, né(e) le {DOB}, diagnostiqué(e) avec {CONDITION} le {DATE}.", "health", "fr"),
    ("Ordonnance pour {NAME}: {DRUG} {DOSE} deux fois par jour.", "health", "fr"),
    ("Résultats d'analyse de {NAME}: glycémie 7.8, cholestérol LDL 3.4.", "health", "fr"),
    ("Note psychiatrique: {NAME} présente {CONDITION}, orienté(e) vers la psychiatrie.", "health", "fr"),
    ("Antécédents chirurgicaux de {NAME}: appendicectomie effectuée le {DATE} au {HOSPITAL}.", "health", "fr"),
    ("Dossier allergie: {NAME} est allergique à {ALLERGEN}, EpiPen prescrit.", "health", "fr"),
    ("Résumé de sortie pour {NAME}: post-opératoire {CONDITION}, prescrit {DRUG} {DOSE}.", "health", "fr"),
    ("Historique médical: {NAME}, né(e) le {DOB} — {CONDITION} chronique depuis 2018.", "health", "fr"),
    ("Admission en soins intensifs: {NAME}, {CONDITION}, admis(e) le {DATE} au {HOSPITAL}.", "health", "fr"),
    # Arabic health
    ("المريض {NAME}، تاريخ الميلاد {DOB}، تم تشخيصه بـ {CONDITION} بتاريخ {DATE}.", "health", "ar"),
    ("وصفة طبية لـ {NAME}: {DRUG} {DOSE} مرتين يومياً.", "health", "ar"),
    ("نتائج التحاليل لـ {NAME}: سكر الدم 7.8، كوليسترول LDL 3.4.", "health", "ar"),
    ("سجل الحساسية: {NAME} لديه حساسية من {ALLERGEN}، تم وصف حقنة أبينفرين.", "health", "ar"),
    ("ملخص الخروج لـ {NAME}: ما بعد الجراحة {CONDITION}، وُصف {DRUG} {DOSE}.", "health", "ar"),
    ("تاريخ طبي: {NAME}، DOB {DOB} — {CONDITION} مزمن منذ 2018.", "health", "ar"),
    ("قيد دخول العناية المركزة: {NAME}، {CONDITION}، الدخول {DATE} في {HOSPITAL}.", "health", "ar"),

    # ══════════════════════════════════════════════ CREDENTIALS ══════
    # English credentials
    ("The production API key is {API_KEY}, keep it strictly confidential.", "credentials", "en"),
    ("{DB_CONN} — do not share this connection string.", "credentials", "en"),
    ("My password is {PASSWORD}, please reset it in the admin panel.", "credentials", "en"),
    ("AWS credentials: access key AKIAIOSFODNN7EXAMPLE, secret wJalrXUtnFEMI/K7MDENG.", "credentials", "en"),
    ("GitHub PAT: {API_KEY} with repo and packages scope.", "credentials", "en"),
    ("JWT secret for {SVC} is hs256-secret-2024-prod-xk9m, rotate quarterly.", "credentials", "en"),
    ("Slack webhook: https://hooks.slack.com/services/T00/B00/xxxx — do not share.", "credentials", "en"),
    ("SSH private key passphrase for prod server: {PASSWORD}.", "credentials", "en"),
    ("Root database password changed to {PASSWORD}, update all services.", "credentials", "en"),
    ("Kubernetes secret for {SVC}: {PASSWORD}. Store in vault immediately.", "credentials", "en"),
    ("OAuth client secret for {SVC}: {API_KEY}. Expires in 90 days.", "credentials", "en"),
    ("Encryption key for prod backups: {PASSWORD}. Back up to cold storage.", "credentials", "en"),
    # French credentials
    ("La clé API de production est {API_KEY}, à ne jamais divulguer.", "credentials", "fr"),
    ("{DB_CONN} — ne partagez pas cette chaîne de connexion.", "credentials", "fr"),
    ("Mon mot de passe est {PASSWORD}, veuillez le réinitialiser dans le panneau admin.", "credentials", "fr"),
    ("Token GitHub: {API_KEY} avec portée repo et packages.", "credentials", "fr"),
    ("Secret JWT pour {SVC}: hs256-secret-2024-prod-xk9m, à renouveler chaque trimestre.", "credentials", "fr"),
    ("Mot de passe root de la base de données changé en {PASSWORD}, mettez à jour tous les services.", "credentials", "fr"),
    ("Clé OAuth pour {SVC}: {API_KEY}. Expire dans 90 jours.", "credentials", "fr"),
    ("Passphrase SSH pour le serveur de production: {PASSWORD}.", "credentials", "fr"),
    ("Secret Kubernetes pour {SVC}: {PASSWORD}. Stockez immédiatement dans le coffre.", "credentials", "fr"),
    # Arabic credentials
    ("مفتاح API للإنتاج هو {API_KEY}، يجب الحفاظ على سريته التامة.", "credentials", "ar"),
    ("{DB_CONN} — لا تشارك سلسلة الاتصال هذه.", "credentials", "ar"),
    ("كلمة المرور الخاصة بي هي {PASSWORD}، يرجى إعادة تعيينها.", "credentials", "ar"),
    ("رمز GitHub: {API_KEY} مع صلاحيات الريبو والحزم.", "credentials", "ar"),
    ("سر JWT لـ {SVC}: hs256-secret-2024-prod-xk9m، يجب تدويره كل ربع سنة.", "credentials", "ar"),
    ("تغيير كلمة مرور الجذر لقاعدة البيانات إلى {PASSWORD}، حدث جميع الخدمات.", "credentials", "ar"),
    ("مفتاح التشفير للنسخ الاحتياطية للإنتاج: {PASSWORD}. خزن في تخزين بارد.", "credentials", "ar"),
]


def generate_dataset(n_per_class_per_lang: int = SAMPLES_PER_CLASS_PER_LANG):
    """
    Generate a perfectly balanced dataset.
    For each (label, language) cell: generate exactly n_per_class_per_lang samples
    by cycling through templates with random slot fills.
    """
    log.info(f"Generating dataset: {n_per_class_per_lang} samples × 6 labels × 3 langs "
             f"= {n_per_class_per_lang * 18} total")

    # Group templates by (label, language)
    by_cell = {}
    for template, label, lang in TEMPLATES:
        key = (label, lang)
        by_cell.setdefault(key, []).append(template)

    texts, labels, langs = [], [], []
    missing = []

    for label in LABELS:
        for lang in ("en", "fr", "ar"):
            key = (label, lang)
            cell_templates = by_cell.get(key, [])
            if not cell_templates:
                missing.append(key)
                continue

            for i in range(n_per_class_per_lang):
                template = cell_templates[i % len(cell_templates)]
                # Add slight variation by sometimes prepending/appending context
                text = fill(template)
                if random.random() < 0.2:
                    text = random.choice([
                        f"Hi, {text}",
                        f"FYI: {text}",
                        f"Please note: {text}",
                        f"{text} Please handle this urgently.",
                        f"{text} Let me know if you need more details.",
                    ])
                texts.append(text.strip())
                labels.append(label)
                langs.append(lang)

    if missing:
        log.warning(f"Missing templates for: {missing}")

    # Shuffle
    combined = list(zip(texts, labels, langs))
    random.shuffle(combined)
    texts, labels, langs = zip(*combined)

    log.info("Distribution:")
    import pandas as pd
    df = pd.DataFrame({"text": texts, "label": labels, "language": langs})
    print(df.groupby(["label", "language"]).size().unstack(fill_value=0).to_string())
    return list(texts), list(labels), list(langs)


# ─── SetFit Training ──────────────────────────────────────────────────────────

def train_setfit(texts, labels):
    """
    Train using SetFit — much faster than full fine-tuning.
    Uses contrastive learning on sentence pairs, then a lightweight classifier head.
    """
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError:
        log.error("SetFit not installed. Run: pip install setfit -q")
        raise

    log.info("Loading SetFit base model...")
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        labels=LABELS,
        multi_target_strategy=None,
    )

    # Build HuggingFace Dataset
    label_ids = [LABEL2ID[l] for l in labels]
    n = len(texts)
    split = int(n * 0.85)

    train_ds = Dataset.from_dict({
        "text":  texts[:split],
        "label": label_ids[:split],
    })
    eval_ds = Dataset.from_dict({
        "text":  texts[split:],
        "label": label_ids[split:],
    })

    log.info(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    args = TrainingArguments(
        batch_size=32,
        num_epochs=2,                   # SetFit only needs 1-2 epochs
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric="accuracy",
    )

    log.info("Starting SetFit training (expected: 8-12 min on Colab GPU)...")
    trainer.train()

    return model, trainer


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, texts, labels, langs):
    from sklearn.metrics import classification_report
    log.info("Evaluating per language...")

    texts_arr  = np.array(texts)
    labels_arr = np.array([LABEL2ID[l] for l in labels])
    langs_arr  = np.array(langs)

    all_preds = model.predict(texts_arr.tolist())
    all_preds = np.array(all_preds)

    for lang in ("en", "fr", "ar"):
        mask = langs_arr == lang
        if mask.sum() == 0:
            continue
        report = classification_report(
            labels_arr[mask], all_preds[mask],
            target_names=LABELS, zero_division=0
        )
        log.info(f"\n=== {lang.upper()} ===\n{report}")


# ─── Save model ───────────────────────────────────────────────────────────────

def save_model(model):
    """Save to both local /content and Google Drive."""
    local_path = LOCAL_DIR / "model_final"
    drive_path = SAVE_DIR / "model_final"

    model.save_pretrained(str(local_path))
    log.info(f"✓ Saved locally → {local_path}")

    try:
        if drive_path.parent.exists():
            shutil.copytree(str(local_path), str(drive_path), dirs_exist_ok=True)
            log.info(f"✓ Saved to Drive → {drive_path}")
    except Exception as e:
        log.warning(f"Drive save failed: {e} — model is safe locally at {local_path}")

    return local_path


# ─── ONNX Export (no optimum needed) ─────────────────────────────────────────

def export_onnx(model_path: Path):
    """
    Export the sentence transformer backbone to ONNX.
    Uses torch.onnx directly — no optimum required.
    """
    log.info("Exporting to ONNX...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        log.error(f"Missing package: {e}")
        return

    onnx_dir  = LOCAL_DIR / "model_onnx"
    quant_dir = LOCAL_DIR / "model_quantized"
    onnx_dir.mkdir(exist_ok=True)
    quant_dir.mkdir(exist_ok=True)

    # The SetFit model wraps a sentence-transformer; get the underlying HF model
    # SetFit stores the backbone as model.model_body
    try:
        backbone_path = model_path / "model_body"
        if not backbone_path.exists():
            backbone_path = model_path  # fallback
        tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
        hf_model  = AutoModel.from_pretrained(str(backbone_path))
    except Exception as e:
        log.warning(f"Could not load backbone directly: {e}")
        log.info("Trying alternative: load from original model name...")
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # Load fine-tuned weights if available
        try:
            hf_model = AutoModel.from_pretrained(str(model_path))
        except:
            hf_model = AutoModel.from_pretrained(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

    hf_model.eval()

    # Dummy input for tracing
    dummy = tokenizer(
        "This is a test sentence for export.",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    onnx_path = onnx_dir / "backbone.onnx"

    with torch.no_grad():
        torch.onnx.export(
            hf_model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    log.info(f"  ONNX fp32: {size_mb:.1f} MB → {onnx_path}")

    # Quantize to INT8
    quant_path = quant_dir / "backbone_quantized.onnx"
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quant_path),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    size_mb_q = quant_path.stat().st_size / 1024 / 1024
    log.info(f"  ONNX INT8: {size_mb_q:.1f} MB → {quant_path}")

    # Save tokenizer to quantized dir
    tokenizer.save_pretrained(str(quant_dir))

    # Also save label config
    config = {"id2label": ID2LABEL, "label2id": LABEL2ID, "max_length": 128}
    with open(quant_dir / "datashield_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy to Drive
    for src, name in [(onnx_dir, "model_onnx"), (quant_dir, "model_quantized")]:
        try:
            dst = SAVE_DIR / name
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            log.info(f"  ✓ Copied {name} to Drive")
        except Exception as e:
            log.warning(f"  Drive copy failed for {name}: {e}")

    return quant_dir


# ─── Smoke test ───────────────────────────────────────────────────────────────

def smoke_test_setfit(model):
    """Quick test of the SetFit model directly (before ONNX)."""
    log.info("SetFit smoke test...")
    cases = [
        ("What is the best way to manage a remote team?",       "safe"),
        ("My SSN is 123-45-6789 and email is alice@corp.com.",  "pii"),
        ("Wire 50,000 EUR to IBAN FR7630006000011234567890189.", "financial"),
        ("API key sk-prod-xK9mN2pL8qR5vY3w — do not share.",    "credentials"),
        ("Patient diagnosed with Type 2 Diabetes on 2024-01-15.","health"),
        ("Project Helios launch Q4 2025, NDA applies.",          "confidential"),
        ("اسمي محمد وبريدي هو m@example.com",                    "pii"),
        ("Mon mot de passe est P@ssw0rd!2024",                   "credentials"),
    ]
    texts = [c[0] for c in cases]
    preds = model.predict(texts)
    print("\n" + "─" * 68)
    print(f"{'TEXT':<44} {'EXPECTED':<14} {'GOT':<10}")
    print("─" * 68)
    for (text, expected), pred_id in zip(cases, preds):
        pred = ID2LABEL[int(pred_id)]
        ok = "✅" if pred == expected else "❌"
        short = (text[:41] + "…") if len(text) > 44 else text
        print(f"{ok} {short:<43} {expected:<14} {pred}")
    print("─" * 68)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("  DataShield AI — Fast Training Pipeline")
    log.info("  Model: paraphrase-multilingual-MiniLM-L12-v2")
    log.info("=" * 55)

    # 1. Generate balanced dataset
    texts, labels, langs = generate_dataset()

    # 2. Train
    model, trainer = train_setfit(texts, labels)

    # 3. Evaluate per language
    evaluate_model(model, texts, labels, langs)

    # 4. Smoke test
    smoke_test_setfit(model)

    # 5. Save
    model_path = save_model(model)

    # 6. ONNX export
    quant_dir = export_onnx(model_path)

    # 7. Final summary
    log.info("\n" + "═" * 55)
    log.info("  DONE")
    log.info(f"  SetFit model  → {model_path}")
    if quant_dir:
        log.info(f"  ONNX INT8     → {quant_dir}")
    log.info(f"  Drive backup  → {SAVE_DIR}")
    log.info("═" * 55)


if __name__ == "__main__":
    main()
