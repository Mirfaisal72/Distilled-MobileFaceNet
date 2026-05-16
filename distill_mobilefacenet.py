import os
import time
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from models.mobilefacenet import MobileFaceNet, ClassificationHead
from utils.data import AlignedFaceDataset, default_transform


def get_teacher():
    """
    Initialize InsightFace teacher using ONNXRuntime backend.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(160, 160))
        return app
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize InsightFace teacher. Install 'insightface' and 'onnxruntime'."
        ) from e


def teacher_embed(app, bimgs: torch.Tensor) -> torch.Tensor:
    """
    Compute teacher embeddings for a batch of images.
    Input bimgs: (B,3,112,112) in [-1,1]. Returns (B, 512) normalized.
    """
    b = bimgs.size(0)
    out = []
    mask = []  # True where teacher found a face, False for zero-embedding fallback
    bimgs_np = ((bimgs.detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    for i in range(b):
        img = np.transpose(bimgs_np[i], (1, 2, 0))  # HWC RGB
        img = img[:, :, ::-1].copy()  # RGB -> BGR (InsightFace expects BGR)
        faces = app.get(img)
        if not faces:
            # fallback: zero embedding (will be masked out of KD loss)
            out.append(np.zeros(512, dtype=np.float32))
            mask.append(False)
        else:
            # use largest face
            face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            emb = face.normed_embedding.astype(np.float32)
            out.append(emb)
            mask.append(True)
    out = torch.tensor(np.stack(out, axis=0))
    out = F.normalize(out, p=2, dim=1)
    mask = torch.tensor(mask, dtype=torch.bool)
    return out, mask


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Dataset
    transform = default_transform()
    dataset = AlignedFaceDataset(args.data_root, transform=transform)
    num_classes = len(dataset.classes)
    if num_classes < 2:
        raise RuntimeError(f"Need at least 2 classes in {args.data_root} for training.")

    # Split train/val
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    student = MobileFaceNet(embedding_size=args.embedding, dropout=args.dropout).to(device)
    head = ClassificationHead(embedding_size=args.embedding, num_classes=num_classes).to(device)

    # Teacher
    teacher = get_teacher()

    # Load checkpoint if requested
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            # Load student weights
            if "student" in checkpoint:
                student.load_state_dict(checkpoint["student"])
            else:
                student.load_state_dict(checkpoint) # fallback if it's just the model state
            
            # Load head weights if available and shapes match
            if "head" in checkpoint and checkpoint["head"]["weight"].shape == head.weight.shape:
                head.load_state_dict(checkpoint["head"])
            print("Resumed successfully.")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Training from scratch.")

    # Optim
    params = list(student.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    # Warmup for first 5 epochs, then cosine decay
    warmup_epochs = min(5, args.epochs // 4)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Losses
    ce_loss = nn.CrossEntropyLoss()

    best_val = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        student.train(); head.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        running = {"kd": 0.0, "ce": 0.0}
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                t_emb, t_mask = teacher_embed(teacher, imgs)
                t_emb = t_emb.to(device)
                t_mask = t_mask.to(device)

            s_emb = student(imgs)
            logits = head(s_emb)

            # Cosine KD loss (only on samples where teacher detected a face)
            if t_mask.any():
                cos_sim = F.cosine_similarity(s_emb[t_mask], t_emb[t_mask], dim=1)
                loss_kd = (1.0 - cos_sim).mean()
            else:
                loss_kd = torch.tensor(0.0, device=device)
            loss_ce = ce_loss(logits, labels)
            loss = args.kd_weight * loss_kd + (1.0 - args.kd_weight) * loss_ce

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            running["kd"] += loss_kd.item()
            running["ce"] += loss_ce.item()
            pbar.set_postfix({"kd": f"{running['kd']/ (pbar.n+1):.3f}", "ce": f"{running['ce']/ (pbar.n+1):.3f}"})

        scheduler.step()

        # Validation (cosine similarity to teacher)
        student.eval(); head.eval()
        sims = []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                t_emb, t_mask = teacher_embed(teacher, imgs)
                t_emb = t_emb.to(device)
                t_mask = t_mask.to(device)
                s_emb = student(imgs)
                if t_mask.any():
                    sims.append(F.cosine_similarity(s_emb[t_mask], t_emb[t_mask], dim=1).mean().item())
        val_metric = float(np.mean(sims))
        tqdm.write(f"Val cosine similarity (student vs teacher): {val_metric:.4f}")

        if val_metric > best_val:
            best_val = val_metric
            torch.save({
                "student": student.state_dict(),
                "head": head.state_dict(),
                "classes": dataset.classes,
            }, os.path.join(args.out_dir, "best.pt"))

    # Optional pruning
    if args.prune:
        tqdm.write("Pruning conv layers (global L1) and fine-tuning...")
        apply_pruning(student, amount=args.prune_amount)
        # brief fine-tune
        for epoch in range(1, 1 + max(1, args.ft_epochs)):
            student.train(); head.train()
            pbar = tqdm(train_loader, desc=f"FT Epoch {epoch}", unit="batch")
            for imgs, labels, _ in pbar:
                imgs = imgs.to(device); labels = labels.to(device)
                with torch.no_grad():
                    t_emb, t_mask = teacher_embed(teacher, imgs)
                    t_emb = t_emb.to(device)
                    t_mask = t_mask.to(device)
                s_emb = student(imgs)
                logits = head(s_emb)
                if t_mask.any():
                    kd = (1.0 - F.cosine_similarity(s_emb[t_mask], t_emb[t_mask], dim=1)).mean()
                else:
                    kd = torch.tensor(0.0, device=device)
                loss = args.kd_weight * kd + (1.0 - args.kd_weight) * ce_loss(logits, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        torch.save({
            "student": student.state_dict(),
            "head": head.state_dict(),
            "classes": dataset.classes,
        }, os.path.join(args.out_dir, "best_pruned.pt"))

    # Export to ONNX
    if args.export_onnx:
        export_onnx(student, args)

    # KNN evaluation
    if args.eval_knn:
        eval_knn(student, dataset, device, args)


def apply_pruning(model: nn.Module, amount: float = 0.3):
    import torch.nn.utils.prune as prune
    modules = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            modules.append((m, "weight"))
    prune.global_unstructured(
        modules,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    # remove prune reparams
    for m, name in modules:
        prune.remove(m, name)


def export_onnx(model: nn.Module, args):
    model.eval()
    dummy = torch.randn(1, 3, 112, 112)
    onnx_path = os.path.join(args.out_dir, "mobilefacenet_student.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    tqdm.write(f"Exported ONNX to {onnx_path}")


def eval_knn(model: nn.Module, dataset: AlignedFaceDataset, device, args):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score
    # Embed all samples
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    embs = []
    labels = []
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(loader, desc="Embedding", unit="batch"):
            imgs = imgs.to(device)
            e = model(imgs).detach().cpu().numpy()
            embs.append(e)
            labels.append(lbls.numpy())
    embs = np.concatenate(embs, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Split to train/val again
    n = len(labels)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    tr_idx, va_idx = idx[:split], idx[split:]
    knn = KNeighborsClassifier(n_neighbors=args.knn_k, metric="cosine")
    knn.fit(embs[tr_idx], labels[tr_idx])
    preds = knn.predict(embs[va_idx])
    f1 = f1_score(labels[va_idx], preds, average="macro")
    tqdm.write(f"KNN macro-F1: {f1:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Distill MobileFaceNet from InsightFace teacher")
    p.add_argument("--data_root", type=str, default="C:/Users/mirfa/Documents/PROJECT/Face_Detection/Dataset/aligned_data")
    p.add_argument("--out_dir", type=str, default="C:/Users/mirfa/Documents/PROJECT/Face_Detection/outputs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--embedding", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--kd_weight", type=float, default=0.7, help="Weight for distillation loss vs CE")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--prune_amount", type=float, default=0.3)
    p.add_argument("--ft_epochs", type=int, default=3)
    p.add_argument("--export_onnx", action="store_true")
    p.add_argument("--eval_knn", action="store_true")
    p.add_argument("--knn_k", type=int, default=3)
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
