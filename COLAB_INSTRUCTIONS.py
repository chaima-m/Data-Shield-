# ╔══════════════════════════════════════════════════════════════════╗
# ║   DataShield AI — Complete Colab Notebook                       ║
# ║   Copy each cell block into a Colab cell and run in order       ║
# ╚══════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies (~2 min)
# ─────────────────────────────────────────────────────────────────────
"""
!pip install setfit sentence-transformers onnx onnxruntime scikit-learn pandas -q
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 2 — Mount Google Drive
# ─────────────────────────────────────────────────────────────────────
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 3 — Clone your repo (or upload the files manually)
# ─────────────────────────────────────────────────────────────────────
"""
!git clone https://github.com/chaima-m/Data-Shield-.git
%cd Data-Shield-
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 4 — Check GPU is available
# ─────────────────────────────────────────────────────────────────────
"""
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ No GPU — go to Runtime > Change runtime type > T4 GPU")
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 5 — Run training (8-12 min with GPU, ~25 min on CPU)
# ─────────────────────────────────────────────────────────────────────
"""
!python datashield_fast.py
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 6 — Export classifier head weights to JSON
# ─────────────────────────────────────────────────────────────────────
"""
!python export_head_weights.py
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 7 — Verify outputs
# ─────────────────────────────────────────────────────────────────────
"""
import os
for root, dirs, files in os.walk('/content/datashield_output'):
    level = root.replace('/content/datashield_output', '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    for f in files:
        size = os.path.getsize(os.path.join(root, f)) / 1024 / 1024
        print(f'{indent}  {f}  ({size:.1f} MB)')
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 8 — Download the quantized model folder
# ─────────────────────────────────────────────────────────────────────
"""
import shutil
shutil.make_archive('/content/datashield_model', 'zip', '/content/datashield_output/model_quantized')
from google.colab import files
files.download('/content/datashield_model.zip')
"""

# ─────────────────────────────────────────────────────────────────────
# WHAT YOU GET IN model_quantized.zip:
# ─────────────────────────────────────────────────────────────────────
#   backbone_quantized.onnx     ← main model (~25-40 MB)
#   datashield_config.json      ← label map + classifier weights
#   vocab.txt                   ← tokenizer vocabulary
#   tokenizer.json              ← full tokenizer config
#   tokenizer_config.json
#
# ─────────────────────────────────────────────────────────────────────
# EXPECTED RESULTS:
# ─────────────────────────────────────────────────────────────────────
#   Training time:    8-12 min (T4 GPU)
#   Model size:       ~25-40 MB quantized
#   Accuracy:         >95% on balanced test set
#   Inference speed:  <50ms per prompt in browser
#   Languages:        EN ✓  FR ✓  AR ✓
