# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from collections import defaultdict

DATASET  = Path("/Users/mac/Detect_Drill_Bit/final-dataset")
CLASSES  = ["Broken", "Chipped", "Scratched", "Severe_Rust", "Tip_Wear"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- Helpers ---
def img_hash(f):
    return hashlib.md5(open(str(f), "rb").read()).hexdigest()

def hash_map(img_dir):
    return {img_hash(f): f for f in Path(img_dir).glob("*") if f.suffix.lower() in IMG_EXTS}

def class_dist(label_dir, name):
    counts = defaultdict(int)
    files  = list(Path(label_dir).glob("*.txt"))
    for lf in files:
        for line in lf.read_text().strip().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    mx = max(counts.values(), default=1)
    print("\n[{}] {} images | {} objects".format(name, len(files), sum(counts.values())))
    for i, cls in enumerate(CLASSES):
        n = counts[i]
        print("  {} {:<14} {:4d}  {}".format(i, cls, n, "#" * int(n / mx * 20)))

# --- Overview ---
train_dir = DATASET / "train" / "images"
val_dir   = DATASET / "val"   / "images"
print("Train: {} | Val: {}".format(len(list(train_dir.glob("*"))), len(list(val_dir.glob("*")))))

# --- Leakage check ---
train_h  = hash_map(train_dir)
val_h    = hash_map(val_dir)
overlap  = set(train_h) & set(val_h)

if overlap:
    print("\n[WARNING] {} anh trung lap train/val:".format(len(overlap)))
    for i, h in enumerate(sorted(overlap), 1):
        print("  [{:02d}] {}".format(i, val_h[h].name))
else:
    print("\n[OK] Khong co leakage")

# --- Class distribution ---
class_dist(DATASET / "train" / "labels", "TRAIN")
class_dist(DATASET / "val"   / "labels", "VAL")

# --- Fix ---
if overlap:
    for h in overlap:
        img = val_h[h]
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        img.unlink()
        if lbl.exists(): lbl.unlink()
    print("Da xoa {}. Val con lai: {} anh".format(len(overlap), len(list(val_dir.glob("*")))))