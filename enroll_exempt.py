#!/usr/bin/env python3
"""
Exempt Person Enrollment
========================
Extracts appearance embeddings from a target person in a video clip.
Saves verification frames so you can confirm the correct person was captured.

Usage:
    python enroll_exempt.py --source bedge.mp4 --start 3 --end 9
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import os, argparse


# ═══════════════════════════════════════════════════════
#  FEATURE EXTRACTOR (shared with bus_passenger_counter)
# ═══════════════════════════════════════════════════════
class PersonFeatureExtractor:
    """MobileNetV2-based feature extractor — outputs a 1280-d embedding."""

    def __init__(self):
        self.model = models.mobilenet_v2(weights="DEFAULT")
        self.model.classifier = nn.Identity()
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return L2-normalised 1280-d vector from a BGR crop."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = self.preprocess(crop_rgb).unsqueeze(0)
        with torch.no_grad():
            feat = self.model(tensor)
        vec  = feat.squeeze().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Enroll exempt person")
    p.add_argument("--source",     default="bedge.mp4",          help="Input video")
    p.add_argument("--start",      type=float, default=3.0,      help="Start time (seconds)")
    p.add_argument("--end",        type=float, default=9.0,      help="End time (seconds)")
    p.add_argument("--model",      default="yolov8s.pt",         help="YOLO weights")
    p.add_argument("--output-dir", default="exempt_enrollment",  help="Verification frames dir")
    p.add_argument("--output-emb", default="exempt_embedding.npy", help="Output embedding file")
    p.add_argument("--sample-every", type=int, default=5,        help="Sample every N frames")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading YOLO for detection…")
    yolo = YOLO(args.model)

    print("Loading feature extractor (MobileNetV2)…")
    extractor = PersonFeatureExtractor()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.source}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(args.start * fps)
    end_frame   = int(args.end   * fps)

    print(f"Video  : {int(cap.get(3))}x{int(cap.get(4))} @ {fps:.1f}fps")
    print(f"Window : {args.start}s – {args.end}s  (frames {start_frame}–{end_frame})")
    print(f"Sampling every {args.sample_every} frames")
    print("─" * 55)

    embeddings  = []
    frame_no    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        if frame_no < start_frame:
            continue
        if frame_no > end_frame:
            break
        if (frame_no - start_frame) % args.sample_every != 0:
            continue

        # ── Detect persons ────────────────────────────────────────────
        results = yolo(frame, classes=[0], conf=0.30, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            continue

        # Pick the LARGEST detection (most prominent person)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best  = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[best].astype(int)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # ── Extract embedding ─────────────────────────────────────────
        emb = extractor.extract(crop)
        embeddings.append(emb)

        # ── Save verification images ──────────────────────────────────
        vis = frame.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        tag = f"Enrollment #{len(embeddings)}"
        cv2.putText(vis, tag, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        full_path = os.path.join(args.output_dir,
                                 f"frame_{frame_no:05d}_full.jpg")
        crop_path = os.path.join(args.output_dir,
                                 f"frame_{frame_no:05d}_crop.jpg")
        cv2.imwrite(full_path, vis)
        cv2.imwrite(crop_path, crop)

        print(f"  Frame {frame_no:5d} | bbox [{x1},{y1},{x2},{y2}] "
              f"| saved embedding #{len(embeddings)}")

    cap.release()

    # ── Save averaged embedding ───────────────────────────────────────
    if embeddings:
        avg = np.mean(embeddings, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        np.save(args.output_emb, avg)

        print("─" * 55)
        print(f"  Samples collected : {len(embeddings)}")
        print(f"  Embedding saved   : {args.output_emb}")
        print(f"  Verification dir  : {args.output_dir}/")
        print("─" * 55)
        print("Check the images in the verification dir to confirm"
              " the correct person was captured.")
    else:
        print("⚠ No persons detected in the specified time range!")


if __name__ == "__main__":
    main()
