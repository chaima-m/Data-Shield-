"""
DataShield — Export SetFit classifier head weights to JSON
===========================================================
Run this AFTER datashield_fast.py finishes training.
It serializes the LogisticRegression head into datashield_config.json
so the browser JS can run classification without a second ONNX model.

In Colab:
    !python export_head_weights.py
"""

import json
import numpy as np
from pathlib import Path

MODEL_PATH = Path("/content/datashield_output/model_final")
QUANT_DIR  = Path("/content/datashield_output/model_quantized")
DRIVE_DIR  = Path("/content/drive/MyDrive/datashield/model_quantized")

LABELS   = ["safe", "pii", "financial", "confidential", "health", "credentials"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def export_head():
    try:
        from setfit import SetFitModel
        import torch
    except ImportError:
        print("pip install setfit torch")
        return

    print("Loading SetFit model...")
    model = SetFitModel.from_pretrained(str(MODEL_PATH))

    head = model.model_head

    config = {
        "id2label":   ID2LABEL,
        "label2id":   LABEL2ID,
        "max_length": 128,
    }

    # Try to get sklearn LogisticRegression weights
    try:
        # SetFit head is a sklearn Pipeline or LogisticRegression
        clf = head
        # Navigate pipeline if needed
        if hasattr(clf, 'steps'):
            clf = clf.steps[-1][1]

        weights = clf.coef_.tolist()    # shape: [n_classes, hidden_size]
        biases  = clf.intercept_.tolist()

        config["weights"] = weights
        config["biases"]  = biases
        print(f"✓ Logistic regression weights: {len(weights)} classes × {len(weights[0])} dims")

    except AttributeError:
        # Fallback: try PyTorch linear head
        try:
            state = head.state_dict()
            w_key = [k for k in state if 'weight' in k]
            b_key = [k for k in state if 'bias' in k]
            if w_key:
                config["weights"] = state[w_key[0]].numpy().tolist()
                config["biases"]  = state[b_key[0]].numpy().tolist() if b_key else [0]*len(LABELS)
                print(f"✓ PyTorch linear weights extracted")
        except Exception as e:
            print(f"Could not extract weights: {e}")
            # Last resort: compute prototype embeddings
            export_prototypes(model, config)

    # Save
    for target in [QUANT_DIR, DRIVE_DIR]:
        try:
            target.mkdir(parents=True, exist_ok=True)
            out_path = target / "datashield_config.json"
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"✓ Saved → {out_path}")
        except Exception as e:
            print(f"  Save to {target} failed: {e}")


def export_prototypes(model, config):
    """
    Fallback: compute mean embeddings per class (prototype network approach).
    These are used in the JS ClassifierHead for cosine similarity classification.
    """
    print("Computing prototype embeddings as fallback...")
    from setfit import SetFitModel

    test_sentences = {
        "safe":         ["What is machine learning?", "How do I center a div?",
                         "Write a Python sort function.", "Explain TCP vs UDP."],
        "pii":          ["My SSN is 123-45-6789.", "Contact Alice at alice@corp.com.",
                         "Employee John Smith, DOB 1985-03-15.", "Send to 42 Oak Street."],
        "financial":    ["Wire 50,000 EUR to IBAN DE89370400440532013000.",
                         "Process card 4111 1111 1111 1111 CVV 737.",
                         "Q3 revenue was 1.2M, projecting 1.5M."],
        "confidential": ["Project Helios launches Q4 under NDA.",
                         "Acquisition target TechCorp valued at 50M. Eyes only.",
                         "Board deck shows 12% revenue miss — do not share."],
        "health":       ["Patient diagnosed with Type 2 Diabetes.",
                         "Prescription: Metformin 500mg twice daily.",
                         "Allergy record: allergic to Penicillin."],
        "credentials":  ["API key sk-prod-xK9mN2pL8qR5vY3w — do not share.",
                         "DB connection: postgresql://admin:pass@db.internal/prod.",
                         "My password is P@ssw0rd!2024."],
    }

    prototypes = {}
    body = model.model_body

    for label, sentences in test_sentences.items():
        embeddings = body.encode(sentences, convert_to_numpy=True)
        proto = embeddings.mean(axis=0)
        # Normalize
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes[label] = proto.tolist()

    config["prototypes"] = [prototypes[l] for l in ["safe", "pii", "financial",
                                                      "confidential", "health", "credentials"]]
    print(f"✓ Computed {len(prototypes)} prototype embeddings")


if __name__ == "__main__":
    export_head()
