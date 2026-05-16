# Distilled MobileFaceNet Pipeline

This workspace adds a distilled MobileFaceNet training pipeline using an InsightFace teacher, pruning, and INT8 TFLite export guidance.

## Overview
- Distillation: Student MobileFaceNet learns from InsightFace (ArcFace) embeddings via MSE on normalized embeddings + CE on class labels.
- Pruning: Global L1 unstructured pruning on conv weights, followed by brief fine-tuning to recover accuracy.
- Export: ONNX export; TFLite INT8 conversion instructions using `onnx2tf` with a representative dataset.
- Eval: KNN on embeddings for identity classification and macro-F1.

## File Map
- models/mobilefacenet.py — student model + cosine classification head
- utils/data.py — aligned dataset loader with augmentation
- distill_mobilefacenet.py — training + distillation + pruning + ONNX export + KNN eval
- requirements.txt — dependencies

## Prerequisites
1. Generate aligned faces into `Dataset/aligned_data` using your existing `align_faces.py`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For InsightFace teacher:
```bash
pip install insightface onnxruntime
```

## Train + Distill
Run training with default paths:
```bash
python distill_mobilefacenet.py --epochs 20 --batch_size 32 --eval_knn --export_onnx
```
Optional pruning and fine-tuning:
```bash
python distill_mobilefacenet.py --prune --prune_amount 0.3 --ft_epochs 3
```

## Convert ONNX → TFLite (INT8)
Use `onnx2tf` with a representative dataset for INT8 quantization:
```bash
pip install onnx2tf tensorflow
onnx2tf -i outputs/mobilefacenet_student.onnx -o outputs/tf_model -oiq -rt "Dataset/aligned_data" --input_shapes "1,112,112,3" --quiet
```
Then convert TF to TFLite:
```bash
python - << 'PY'
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('outputs/tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('outputs/mobilefacenet_student_int8.tflite', 'wb').write(tflite_model)
PY
```
Note: For full INT8 (weights + activations), provide a representative dataset function. See TensorFlow docs.

## KNN Classification
After training, evaluate KNN macro-F1:
```bash
python distill_mobilefacenet.py --eval_knn --cpu
```

## Tips
- Adjust `--kd_weight` to balance distillation vs CE classification.
- Increase `--epochs` for larger datasets; monitor val cosine similarity.
- For edge devices, prefer pruned + INT8 models to reduce latency and power.
