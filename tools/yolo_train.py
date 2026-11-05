#!/usr/bin/env python3
"""yolo_train.py - fine-tune a YOLO segmentation model on the synthetic dataset.

    python3 tools/yolo_dataset.py --out datasets/so101_seg --images 6000
    python3 tools/yolo_train.py --data datasets/so101_seg/data.yaml
    # -> ros2_ws/src/so101_perception/models/so101_seg.pt (used by vision.YoloSegmenter)

Starts from the pretrained COCO weights (transfer learning), images at the camera
resolution (320). Default model: yolo26n-seg (nano, real time on CPU).
Needs ultralytics (AGPL-3.0).
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "ros2_ws/src/so101_perception/models/so101_seg.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="cpu, mps, 0 (cuda)... default: auto")
    ap.add_argument("--project", default=str(ROOT / "runs/yolo"))
    ap.add_argument("--name", default="so101_seg", help="run name (sub-directory of --project)")
    ap.add_argument("--patience", type=int, default=20, help="early stop after N epochs without progress")
    ap.add_argument("--out", default=str(DEST))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, imgsz=320, epochs=args.epochs, batch=args.batch, device=args.device,
                project=args.project, name=args.name, exist_ok=True, patience=args.patience,
                # colours are already randomized in the dataset: mild extra colour augment,
                # default geometric augment
                hsv_h=0.02, hsv_s=0.5, hsv_v=0.5, mosaic=0.5, plots=False, verbose=False)
    best = Path(model.trainer.best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    print(f"best weights: {best} -> {args.out}")
    print(model.val(data=args.data, imgsz=320, plots=False, verbose=False).seg.map50)


if __name__ == "__main__":
    main()
