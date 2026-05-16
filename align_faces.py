import os
import glob
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import numpy as np
import torch
import cv2

SRC_ROOT = "C:/Users/mirfa/Documents/PROJECT/Face_Detection/Dataset/Raw_data"         
DST_ROOT = "C:/Users/mirfa/Documents/PROJECT/Face_Detection/Dataset/aligned_data"     

IMAGE_SIZE = 112             
MARGIN = 10                   
KEEP_ALL = False              
MIN_FACE_SIZE = 20            
SAVE_FORMAT = "PNG"           

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def pil_from_tensor(tensor, enhance=True):
    """
    Convert MTCNN output tensor -> PIL RGB image correctly.
    Handles:
      - torch.Tensor (C,H,W) with ranges: [-1,1], [0,1], or [0,255]
      - numpy arrays in CHW or HWC
    If enhance=True, apply CLAHE auto-brightening (useful for dark crops).
    """
    # get numpy array
    if torch.is_tensor(tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    # If CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1,3,4):
        arr = np.transpose(arr, (1, 2, 0))

    # If grayscale HxW, make 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    # Drop alpha channel if present
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    arr = np.ascontiguousarray(arr)

    # Determine numeric range & scale to 0..255
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))

    # Case: MTCNN returned normalized [-1,1]
    if a_min >= -1.5 and a_max <= 1.5 and a_min < 0:
        arr = ( (arr + 1.0) / 2.0 ) * 255.0
    # Case: floats in [0,1]
    elif np.issubdtype(arr.dtype, np.floating) and a_max <= 1.5:
        arr = arr * 255.0
    # Else assume already 0..255 (or integers) — leave as is

    # Finalize -> uint8
    arr = np.clip(np.round(arr), 0, 255).astype('uint8')

    # Optional enhancement for dark images
    if enhance:
        # arr is RGB uint8
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(arr, mode='RGB')



def main():
    # create MTCNN detector
    mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=MARGIN, select_largest=True, keep_all=KEEP_ALL)

    ensure_dir(DST_ROOT)

    people = [d for d in sorted(os.listdir(SRC_ROOT)) if os.path.isdir(os.path.join(SRC_ROOT, d))]
    if not people:
        print(f"No person folders found in {SRC_ROOT}. Create folders like {SRC_ROOT}/alice/*.jpg")
        return

    for person in people:
        in_dir = os.path.join(SRC_ROOT, person)
        out_dir = os.path.join(DST_ROOT, person)
        ensure_dir(out_dir)

        files = sorted([f for f in glob.glob(os.path.join(in_dir, "*")) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))])
        if not files:
            print(f"  No image files for {person}, skipping.")
            continue

        for fpath in tqdm(files, desc=f"Aligning {person}", unit="img"):
            try:
                img = Image.open(fpath).convert("RGB")
            except Exception as e:
                # skip unreadable files
                print(f"    skip {fpath}: cannot open ({e})")
                continue

            # mtcnn returns a PIL Image or torch Tensor depending on version/config
            try:
                face = mtcnn(img)   # returns a cropped aligned face (as PIL Image or tensor or list if keep_all)
            except Exception as e:
                print(f"    mtcnn error for {fpath}: {e}")
                continue

            if face is None:
                # no face detected
                # you can optionally save original or log filename for manual checking
                # for now, skip
                continue

            # if keep_all=False, face is a single tensor or PIL; if keep_all=True, it is list of faces
            if isinstance(face, list):
                # pick the first (largest) face
                face_img = face[0]
            else:
                face_img = face

            # face_img might be a torch Tensor -> convert to PIL
            if not isinstance(face_img, Image.Image):
                face_img = pil_from_tensor(face_img)
                arr = np.asarray(face_img)
                face_img = Image.fromarray(arr)

            # final safety: resize exactly to IMAGE_SIZE
            face_img = face_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

            out_name = os.path.splitext(os.path.basename(fpath))[0] + "." + SAVE_FORMAT.lower()
            out_path = os.path.join(out_dir, out_name)
            face_img.save(out_path, format=SAVE_FORMAT)
    print("Done. Aligned images saved to:", DST_ROOT)
    print("tensor min/max/dtype/shape:", face.min().item(), face.max().item(), face.dtype, tuple(face.shape))

if __name__ == "__main__":
    main()

