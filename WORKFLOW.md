# Project Workflow: Distilled MobileFaceNet for Image Recognition

This document explains how the files in this workspace work together to build a lightweight, high-accuracy face recognition pipeline using distillation, pruning, and quantization for edge deployment.

---

## Overview
- Preprocess faces to an aligned dataset.
- Train a student MobileFaceNet via distillation from an InsightFace teacher (ArcFace-style embeddings).
- Apply pruning and short fine-tuning to reduce compute without losing accuracy.
- Export the student to ONNX and convert to INT8 TFLite for edge devices.
- Evaluate identity classification via KNN on the learned embeddings with macro-F1.

---

## File Roles

### Data Preparation
- **align_faces.py**: Detects and aligns faces from raw images per person into `Dataset/aligned_data/<person>`. Uses `MTCNN` with margin and optional brightness enhancement. This produces 112×112 PNG crops suitable for model training.

### Model Definition
- **models/mobilefacenet.py**:
  - `MobileFaceNet`: Lightweight CNN optimized for 112×112 inputs. Outputs a 128-D L2-normalized embedding. Includes dropout for regularization.
  - `ClassificationHead`: Cosine classifier (normalized weights) to produce class logits during supervised training.

### Dataset + Augmentation
- **utils/data.py**:
  - `AlignedFaceDataset`: Loads aligned images from `Dataset/aligned_data` as `(image_tensor, label, path)`, expecting folder-per-class layout.
  - `default_transform()`: Augmentations (color jitter, random grayscale, horizontal flip, slight rotation) and normalization to [-1, 1]. Helps reduce overfitting and improve robustness.

### Training / Distillation / Pruning / Export / Eval
- **distill_mobilefacenet.py**:
  - Initializes `AlignedFaceDataset`, splits into train/val, and builds `DataLoader`s.
  - Creates the student `MobileFaceNet` and the cosine `ClassificationHead`.
  - Sets up the teacher via `insightface` `FaceAnalysis`. `teacher_embed()` generates normalized 512-D teacher embeddings per image.
  - Losses:
    - Distillation: MSE between student embeddings and teacher embeddings (both normalized).
    - Supervision: Cross-entropy over identity classes using the cosine classifier.
    - Blend: `loss = kd_weight * MSE + (1 - kd_weight) * CE` (default kd_weight=0.7).
  - Optimization: `AdamW` + cosine LR scheduler, gradient clipping.
  - Validation: Computes mean cosine similarity between student and teacher embeddings; saves best checkpoint.
  - Pruning (optional): Applies global L1 unstructured pruning on all conv weights, then runs brief fine-tuning epochs to recover accuracy; saves pruned checkpoint.
  - Export (optional): Exports the student model to ONNX with dynamic batch axis.
  - KNN Eval (optional): Embeds the full dataset, splits 80/20, trains a KNN classifier (cosine metric), and reports macro-F1.

### Project Metadata
- **requirements.txt**: Python dependencies to install for training, teacher, export, and evaluation.
- **README.md**: Quick setup, commands, and conversion instructions.

---

## End-to-End Workflow

1. **Align Faces** (once per dataset update)
   - Run `align_faces.py` to create `Dataset/aligned_data/<person>/*.png` from raw photos.
   - Ensure at least 2 persons/classes with multiple images each.

2. **Install Dependencies**
   - `pip install -r requirements.txt`
   - Teacher backend: `pip install insightface onnxruntime`

3. **Train + Distill**
   - Example:
     - `python distill_mobilefacenet.py --epochs 20 --batch_size 32 --eval_knn --export_onnx`
   - The script will:
     - Load augmented batches.
     - Compute teacher embeddings using InsightFace.
     - Optimize student to match teacher embeddings while also classifying identities.
     - Save `outputs/best.pt` when validation similarity improves.

4. **Prune + Fine-Tune (Optional)**
   - Example:
     - `python distill_mobilefacenet.py --prune --prune_amount 0.3 --ft_epochs 3`
   - Reduces parameters and compute; fine-tune recovers accuracy. Saves `outputs/best_pruned.pt`.

5. **Export to ONNX**
   - Pass `--export_onnx` during training to write `outputs/mobilefacenet_student.onnx` once done.

6. **Convert to INT8 TFLite (Edge)**
   - Follow `README.md`: use `onnx2tf` to get a TensorFlow model, then TF Lite conversion with optimizations.
   - For full INT8 (weights + activations), provide a representative dataset function (calibration).

7. **Evaluate with KNN**
   - Pass `--eval_knn` to compute embeddings and macro-F1 using a KNN classifier.
   - Good for quick identity classification without training a heavy head.

---

## Key Design Choices
- **Embedding Normalization**: Student outputs are L2-normalized for stable metric learning and cosine classifiers.
- **Augmentation + Dropout**: Reduces overfitting and improves generalization to unseen subjects.
- **Teacher Guidance**: Using InsightFace embeddings (ArcFace) stabilizes training and improves F1 over a baseline small model.
- **Pruning**: Global L1 unstructured pruning targets low-magnitude weights; short fine-tuning retains embedding quality.
- **Quantization**: INT8 TFLite reduces latency and power on mobile/edge hardware with minimal impact on embedding quality.

---

## Configuration Notes
- `--kd_weight`: Balance between distillation and supervised CE (default 0.7). Increase for stronger teacher matching; decrease to rely more on labels.
- `--embedding`: Student embedding size (default 128). Smaller sizes reduce memory and bandwidth.
- `--dropout`: Regularization in the embedding stem (default 0.3).
- `--prune_amount`: Fraction of weights pruned globally in conv layers (default 0.3).

---

## Troubleshooting
- **Teacher not found**: Install `insightface` and `onnxruntime`. Ensure GPU or CPU context is available.
- **No faces detected in teacher**: The teacher returns a zero embedding fallback. Verify your aligned images are centered, clear, and sized 112×112.
- **Dataset structure**: Must be `Dataset/aligned_data/<class>/<image files>`. Each class should have multiple images.
- **ONNX export issues**: Check PyTorch and opset versions; fall back to opset 12 if needed.
- **Windows TFLite conversion**: `onnx2tf` may need WSL or a Linux/macOS env; alternatively export to TorchScript or ONNX Runtime Mobile.

---

## Next Steps
- Log F1 improvements across pruning levels and quantization settings.
- Integrate a margin-based head (ArcFace/CosFace) for label supervision if desired.
- Add a small representative dataset function in TF Lite conversion for improved INT8 calibration.
