import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from models.mobilefacenet import MobileFaceNet

# ── Config ──────────────────────────────────────────────
CHECKPOINT = "C:/Users/mirfa/Documents/PROJECT/Face_Detection/outputs/best.pt"
GALLERY_DIR = "C:/Users/mirfa/Documents/PROJECT/Face_Detection/Dataset/aligned_data"
EMBEDDING_SIZE = 512
DROPOUT = 0.2
CONFIDENCE_THRESHOLD = 0.40  # cosine similarity threshold (0.55 is a good starting point)
IMAGE_SIZE = 112

# ── Inference transform (NO augmentation, just resize + normalize) ──
infer_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ── Load model ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location=device)
classes = checkpoint["classes"]

student = MobileFaceNet(embedding_size=EMBEDDING_SIZE, dropout=DROPOUT).to(device)
student.load_state_dict(checkpoint["student"])
student.eval()

print(f"Loaded model with {len(classes)} classes: {classes}")

# ── Build gallery: average embedding per person ─────────
print("Building face gallery from aligned data...")
gallery_names = []     # list of person names
gallery_embeddings = [] # list of mean embeddings (one per person)

for person in sorted(os.listdir(GALLERY_DIR)):
    person_dir = os.path.join(GALLERY_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    person_embs = []
    for fname in os.listdir(person_dir):
        fpath = os.path.join(person_dir, fname)
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        img = Image.open(fpath).convert("RGB")
        tensor = infer_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = student(tensor)
        person_embs.append(emb)

    if person_embs:
        # Average all embeddings for this person, then L2-normalize
        mean_emb = torch.cat(person_embs, dim=0).mean(dim=0, keepdim=True)
        mean_emb = F.normalize(mean_emb, p=2, dim=1)
        gallery_names.append(person)
        gallery_embeddings.append(mean_emb)
        print(f"  {person}: {len(person_embs)} images -> 1 gallery embedding")

# Stack into (num_people, embedding_size)
gallery_matrix = torch.cat(gallery_embeddings, dim=0)  # (N, 512)
print(f"Gallery ready: {len(gallery_names)} people\n")

# ── Face detector ───────────────────────────────────────
mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=10,
    keep_all=True,
    min_face_size=40,
    device=device,
)

# ── Webcam loop ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit(1)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(rgb)

    if boxes is not None:
        for i, box in enumerate(boxes):
            if probs[i] is None or probs[i] < 0.9:
                continue

            x1, y1, x2, y2 = [int(b) for b in box]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_crop = rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Convert crop to PIL, apply same transform as gallery
            face_pil = Image.fromarray(face_crop)
            face_tensor = infer_transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                live_emb = student(face_tensor)  # (1, 512)

            # Compare against all gallery embeddings
            sims = F.cosine_similarity(live_emb, gallery_matrix, dim=1)  # (N,)
            best_idx = sims.argmax().item()
            best_score = sims[best_idx].item()

            if best_score >= CONFIDENCE_THRESHOLD:
                name = gallery_names[best_idx]
                color = (0, 255, 0)  # green
            else:
                name = "Unknown"
                color = (0, 0, 255)  # red

            label = f"{name} ({best_score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition - Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
