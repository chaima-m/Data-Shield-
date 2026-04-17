"""
DataShield AI — Full Fine-Tuning Pipeline
==========================================
Covers:
  1. Loading & splitting the merged dataset
  2. Tokenization with multilingual DistilBERT / XLM-RoBERTa
  3. Training with Trainer API (weighted loss for class imbalance)
  4. Per-language evaluation
  5. ONNX export + INT8 quantization
  6. Final smoke-test of the quantized model

Requirements:
    pip install transformers datasets torch scikit-learn evaluate \
                onnx onnxruntime optimum[onnxruntime] accelerate

Usage:
    python 2_finetune.py                          # default: distilbert-base-multilingual-cased
    python 2_finetune.py --model xlm-roberta-base # use XLM-R if Arabic F1 < 0.80
    python 2_finetune.py --epochs 3 --batch 32    # quick run
"""

import os
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from evaluate import load as load_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./model_output")
ONNX_DIR   = Path("./model_onnx")
QUANT_DIR  = Path("./model_onnx_quantized")

for d in (OUTPUT_DIR, ONNX_DIR, QUANT_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABEL2ID = {
    "safe": 0,
    "pii": 1,
    "financial": 2,
    "confidential": 3,
    "health": 4,
    "credentials": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

RANDOM_SEED = 42
MAX_LENGTH  = 256   # tokens


# ─── 1. Dataset Loading ────────────────────────────────────────────────────────

def load_and_split(csv_path: Path, test_size: float = 0.1, val_size: float = 0.1):
    """
    Load CSV, stratified-split into train / val / test.
    Returns HuggingFace DatasetDict.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["label"].isin(LABEL2ID)].copy()
    df = df[df["text"].str.len() >= 10].copy()
    df["label_id"] = df["label"].map(LABEL2ID)

    log.info(f"Loaded {len(df)} samples from {csv_path}")
    log.info("Label distribution:\n" + df["label"].value_counts().to_string())

    # Split: first off test, then val from remaining train
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=RANDOM_SEED
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label_id"], random_state=RANDOM_SEED
    )

    log.info(f"Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")

    def to_hf(subset: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            "text":     subset["text"].tolist(),
            "labels":   subset["label_id"].tolist(),
            "language": subset["language"].tolist(),
        })

    return DatasetDict({"train": to_hf(train), "validation": to_hf(val), "test": to_hf(test)})


# ─── 2. Tokenization ──────────────────────────────────────────────────────────

def build_tokenize_fn(tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,          # dynamic padding via DataCollator
            max_length=MAX_LENGTH,
        )
    return tokenize


# ─── 3. Weighted Loss Trainer ─────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies per-class weights to the cross-entropy loss.
    This handles class imbalance without down-sampling.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ─── 4. Metrics ───────────────────────────────────────────────────────────────

_f1_metric  = load_metric("f1")
_acc_metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1  = _f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    acc = _acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    return {"f1": f1, "accuracy": acc}


# ─── 5. Per-language evaluation ───────────────────────────────────────────────

def evaluate_per_language(trainer: Trainer, test_ds: Dataset):
    """Run inference on test set and print classification report per language."""
    log.info("Running per-language evaluation …")
    pred_output = trainer.predict(test_ds)
    preds  = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids
    langs  = test_ds["language"]

    results = {}
    for lang in ("en", "fr", "ar"):
        mask = np.array([l == lang for l in langs])
        if mask.sum() == 0:
            continue
        report = classification_report(
            labels[mask], preds[mask],
            target_names=list(LABEL2ID.keys()),
            zero_division=0,
            output_dict=True,
        )
        results[lang] = report
        log.info(f"\n=== Language: {lang.upper()} ===")
        print(classification_report(
            labels[mask], preds[mask],
            target_names=list(LABEL2ID.keys()),
            zero_division=0,
        ))

    # Save full report
    with open(OUTPUT_DIR / "per_language_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    # Check Arabic F1
    ar_f1 = results.get("ar", {}).get("weighted avg", {}).get("f1-score", 0)
    if ar_f1 < 0.78:
        log.warning(
            f"⚠️  Arabic weighted F1 = {ar_f1:.3f} (target: >0.80). "
            "Consider adding more Arabic training data or switching to XLM-RoBERTa."
        )
    return results


# ─── 6. ONNX Export ───────────────────────────────────────────────────────────

def export_to_onnx(model_dir: Path, onnx_dir: Path, quant_dir: Path):
    """
    Export the fine-tuned model to ONNX and quantize to INT8.
    Uses HuggingFace Optimum library.
    """
    log.info("Exporting to ONNX …")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    # Export (fp32 ONNX)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), export=True
    )
    ort_model.save_pretrained(str(onnx_dir))
    log.info(f"  ONNX model saved → {onnx_dir}")

    # Quantize to INT8 (dynamic quantization — no calibration data needed)
    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    # Fallback config for non-ARM
    try:
        quantizer.quantize(save_dir=str(quant_dir), quantization_config=qconfig)
    except Exception:
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=str(quant_dir), quantization_config=qconfig)

    # Report model sizes
    for p in onnx_dir.glob("*.onnx"):
        size_mb = p.stat().st_size / 1024 / 1024
        log.info(f"  {p.name}: {size_mb:.1f} MB (fp32)")
    for p in quant_dir.glob("*.onnx"):
        size_mb = p.stat().st_size / 1024 / 1024
        log.info(f"  {p.name}: {size_mb:.1f} MB (INT8 quantized)")

    log.info(f"  Quantized model → {quant_dir}")


# ─── 7. Smoke Test ────────────────────────────────────────────────────────────

def smoke_test_onnx(quant_dir: Path):
    """
    Quick sanity check: run 6 test sentences (one per label × languages)
    through the quantized ONNX model.
    """
    log.info("Running smoke test on quantized ONNX model …")

    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))

    onnx_files = list(quant_dir.glob("model_quantized.onnx"))
    if not onnx_files:
        onnx_files = list(quant_dir.glob("*.onnx"))
    if not onnx_files:
        log.warning("No ONNX file found for smoke test.")
        return

    session = ort.InferenceSession(str(onnx_files[0]))

    test_cases = [
        ("What is the best way to manage a remote team?", "safe", "en"),
        ("My SSN is 123-45-6789 and email is john@corp.com.", "pii", "en"),
        ("Virement de 50 000 EUR vers IBAN FR7630006000011234567890189.", "financial", "fr"),
        ("The production API key is sk-prod-xK9mN2pL8qR5vY3w.", "credentials", "en"),
        ("المريض يوسف بنعلي تشخيص مرض السكري من النوع الثاني.", "health", "ar"),
        ("Project Helios launch is Q4 2025, under NDA. Do not share.", "confidential", "en"),
    ]

    print("\n" + "─" * 60)
    print(f"{'TEXT':<45} {'EXPECTED':<12} {'PREDICTED':<12} {'CONF':>6}")
    print("─" * 60)

    for text, expected, lang in test_cases:
        enc = tokenizer(
            text, return_tensors="np",
            truncation=True, padding="max_length", max_length=MAX_LENGTH
        )
        inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = session.run(None, inputs)[0]
        probs  = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        pred_id = int(probs.argmax())
        pred_label = ID2LABEL[pred_id]
        confidence = float(probs[0, pred_id])
        match = "✅" if pred_label == expected else "❌"
        short_text = (text[:42] + "…") if len(text) > 45 else text
        print(f"{match} {short_text:<44} {expected:<12} {pred_label:<12} {confidence:.2f}")

    print("─" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    log.info(f"Using model: {args.model}")
    log.info(f"Epochs: {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    csv_path = DATA_DIR / "datashield_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run 1_merge_datasets.py first."
        )

    dataset = load_and_split(csv_path)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenize_fn = build_tokenize_fn(tokenizer)

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=["text", "language"]
    )
    tokenized.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer)

    # ── Class weights ─────────────────────────────────────────────────────────
    train_labels = np.array(dataset["train"]["labels"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=train_labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    log.info("Class weights: " + str({ID2LABEL[i]: f"{w:.3f}" for i, w in enumerate(class_weights)}))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),      # auto fp16 on GPU
        dataloader_num_workers=0,            # safe for Windows
        seed=RANDOM_SEED,
        report_to="none",                    # disable wandb/tensorboard unless configured
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("Starting training …")
    trainer.train()

    # ── Save best model ───────────────────────────────────────────────────────
    final_model_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    log.info(f"Best model saved → {final_model_dir}")

    # ── Per-language evaluation ───────────────────────────────────────────────
    evaluate_per_language(trainer, tokenized["test"])

    # ── ONNX export ───────────────────────────────────────────────────────────
    if not args.skip_onnx:
        export_to_onnx(final_model_dir, ONNX_DIR, QUANT_DIR)
        # Copy tokenizer files to quantized dir for browser use
        tokenizer.save_pretrained(str(QUANT_DIR))
        smoke_test_onnx(QUANT_DIR)
    else:
        log.info("Skipping ONNX export (--skip-onnx flag set)")

    log.info("\n✅ Pipeline complete.")
    log.info(f"   Fine-tuned model  → {final_model_dir}")
    log.info(f"   ONNX (fp32)       → {ONNX_DIR}")
    log.info(f"   ONNX (INT8 quant) → {QUANT_DIR}  ← use this in the browser")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataShield Fine-Tuning Pipeline")
    parser.add_argument(
        "--model",
        default="distilbert-base-multilingual-cased",
        choices=[
            "distilbert-base-multilingual-cased",
            "xlm-roberta-base",
            "bert-base-multilingual-cased",
        ],
        help="Base model to fine-tune",
    )
    parser.add_argument("--epochs",    type=int,   default=5,    help="Training epochs")
    parser.add_argument("--batch",     type=int,   default=16,   help="Per-device batch size")
    parser.add_argument("--lr",        type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--skip-onnx", action="store_true",      help="Skip ONNX export step")
    args = parser.parse_args()
    main(args)
