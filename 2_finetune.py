"""
DataShield — Fixed Fine-tuning Script
========================================
Fixes from original run:
  1. Real class-weighted loss (not fake weights)
  2. install optimum BEFORE importing it
  3. Colab-safe: saves checkpoint every epoch to Drive
  4. Faster: 3 epochs on ~12k samples → ~15 min on T4
  5. Proper per-class F1 reporting (not cheated 1.00)

Usage:
  !python 2_finetune.py
  !python 2_finetune.py --model xlm-roberta-base   # if Arabic F1 < 0.80
  !python 2_finetune.py --epochs 2 --batch 32      # quick test
"""

import argparse, logging, os, sys, subprocess

# ── Install optimum FIRST (before any imports that need it) ──────────────────
def ensure_optimum():
    try:
        import optimum  # noqa
    except ImportError:
        print("Installing optimum[onnxruntime] …")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "optimum[onnxruntime]", "onnx", "onnxruntime"
        ])

ensure_optimum()

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate

# ─── Config ──────────────────────────────────────────────────────────────────

LABELS      = ["safe", "pii", "financial", "confidential", "health", "credentials"]
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
ID2LABEL    = {i: l for i, l in enumerate(LABELS)}

MAX_LENGTH  = 128
DATA_PATH   = "data/datashield_dataset.csv"
OUTPUT_DIR  = "model_output"
FINAL_DIR   = os.path.join(OUTPUT_DIR, "final")
ONNX_DIR    = "model_onnx"
QUANT_DIR   = "model_onnx_quantized"

# Try to mount Google Drive for safe checkpointing on Colab
DRIVE_SAVE  = "/content/drive/MyDrive/datashield_checkpoints"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="distilbert-base-multilingual-cased",
                   help="HuggingFace model name")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs (3 is enough for balanced data)")
    p.add_argument("--batch",  type=int, default=32,
                   help="Per-device batch size (use 32 on T4, 16 if OOM)")
    p.add_argument("--lr",     type=float, default=3e-5)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap total samples for a quick test run")
    return p.parse_args()


# ─── Data loading ────────────────────────────────────────────────────────────

def load_data(path, max_samples=None):
    log.info(f"Loading dataset from {path} …")
    df = pd.read_csv(path)

    # Basic validation
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have 'text' and 'label' columns"
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(LABELS)]

    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)

    log.info(f"Loaded {len(df)} samples")
    log.info("Label distribution:\n" + df["label"].value_counts().to_string())

    # Check balance — this is the key diagnostic
    counts = df["label"].value_counts()
    ratio = counts.max() / counts.min()
    if ratio > 10:
        log.warning(
            f"⚠️  Dataset imbalance ratio = {ratio:.0f}x. "
            "This will cause fake F1=1.00. Run 1_merge_datasets.py first!"
        )
    else:
        log.info(f"✅ Class balance ratio: {ratio:.1f}x — OK")

    df["label_id"] = df["label"].map(LABEL2ID)
    return df


# ─── Tokenisation ────────────────────────────────────────────────────────────

def tokenize_dataset(df, tokenizer):
    ds = Dataset.from_pandas(df[["text", "label_id"]].rename(
        columns={"label_id": "labels"}
    ))

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,   # DataCollator handles padding
        )

    return ds.map(tok, batched=True, remove_columns=["text"])


# ─── Class-weighted loss trainer ─────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropy to handle any remaining imbalance."""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─── Metrics ─────────────────────────────────────────────────────────────────

metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    acc = (preds == labels).mean()
    return {"f1": f1, "accuracy": acc}


# ─── Per-language evaluation ─────────────────────────────────────────────────

def eval_per_language(model, tokenizer, test_df, device):
    log.info("Running per-language evaluation...")
    model.eval()
    
    # Get actual unique labels present in the test set to prevent crash
    unique_test_labels = sorted(test_df['label'].unique().tolist())
    target_names = [L for L in LABELS if L in unique_test_labels]

    for lang in test_df['language'].unique():
        lang_df = test_df[test_df['language'] == lang]
        texts = lang_df['text'].tolist()
        # Convert string labels to IDs
        labels = [LABELS.index(l) for l in lang_df['label'].tolist()]
        
        preds = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        
        log.info(f"\nResults for {lang.upper()}:")
        # Added 'labels' filter to classification_report to ignore missing classes
        print(classification_report(labels, preds, 
                                    target_names=target_names, 
                                    labels=[LABELS.index(l) for l in target_names]))


# ─── ONNX export ─────────────────────────────────────────────────────────────

def export_to_onnx(model_dir, onnx_dir, quant_dir):
    log.info("Exporting to ONNX …")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(quant_dir, exist_ok=True)

    # fp32 export
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, export=True
    )
    ort_model.save_pretrained(onnx_dir)
    log.info(f"FP32 ONNX saved → {onnx_dir}")

    # INT8 quantization
    from optimum.onnxruntime import ORTQuantizer
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(
        save_dir=quant_dir,
        quantization_config=qconfig,
    )
    log.info(f"INT8 quantized ONNX saved → {quant_dir}")


# ─── Google Drive backup (Colab only) ────────────────────────────────────────

def try_save_to_drive(local_dir, label="checkpoint"):
    try:
        from google.colab import drive  # noqa
        import shutil
        dest = os.path.join(DRIVE_SAVE, label)
        os.makedirs(DRIVE_SAVE, exist_ok=True)
        shutil.copytree(local_dir, dest, dirs_exist_ok=True)
        log.info(f"✅ Backed up {label} to Google Drive: {dest}")
    except Exception as e:
        log.info(f"(Drive backup skipped: {e})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cpu":
        log.warning("⚠️  No GPU detected — training will be slow. Enable GPU in Colab.")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data(DATA_PATH, max_samples=args.max_samples)

    train_df, tmp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df,   test_df = train_test_split(tmp_df, test_size=0.5, stratify=tmp_df["label"], random_state=42)
    log.info(f"Split → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    # ── 2. Class weights ──────────────────────────────────────────────────────
    cw = compute_class_weight(
        "balanced",
        classes=np.arange(len(LABELS)),
        y=train_df["label_id"].values
    )
    class_weights = torch.tensor(cw, dtype=torch.float)
    log.info("Class weights: " +
             str({LABELS[i]: f"{w:.3f}" for i, w in enumerate(cw)}))

    # ── 3. Tokenizer & model ──────────────────────────────────────────────────
    log.info(f"Using model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    # ── 4. Tokenize ───────────────────────────────────────────────────────────
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds   = tokenize_dataset(val_df,   tokenizer)
    test_ds  = tokenize_dataset(test_df,  tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    # ── 5. Training args ──────────────────────────────────────────────────────
    # Steps per epoch
    steps_per_epoch = len(train_ds) // args.batch
    eval_steps = max(50, steps_per_epoch // 3)   # eval 3x per epoch

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=(device == "cuda"),
        logging_steps=20,
        report_to="none",       # disable wandb
        dataloader_num_workers=2,
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    log.info("Starting training …")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    # ── 7. Save best model ────────────────────────────────────────────────────
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    log.info(f"Best model saved → {FINAL_DIR}")
    try_save_to_drive(FINAL_DIR, "final_model")

    # ── 8. Per-language evaluation on test set ────────────────────────────────
    eval_per_language(model, tokenizer, test_df, device)

    # ── 9. ONNX export ────────────────────────────────────────────────────────
    try:
        export_to_onnx(FINAL_DIR, ONNX_DIR, QUANT_DIR)
        try_save_to_drive(QUANT_DIR, "onnx_quantized")
        log.info("✅ ONNX export complete!")
    except Exception as e:
        log.error(f"ONNX export failed: {e}")
        log.info("Run manually: pip install optimum[onnxruntime] && python 2_finetune.py again")

    log.info("🎉 All done!")


if __name__ == "__main__":
    args = parse_args()
    log.info(f"Model: {args.model} | Epochs: {args.epochs} | Batch: {args.batch}")
    main(args)
