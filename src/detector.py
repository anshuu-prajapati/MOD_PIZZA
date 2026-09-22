"""Person detection + anonymous tracking.

ByteTrack is appearance-free: IDs are assigned by motion/IoU only and are reset
for every camera and every run. They carry no identity and cannot be matched to
a person across cameras, days, or visits. That is deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PERSON_CLASS = 0


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    conf: float
    foot: tuple[float, float]


class PersonTracker:
    def __init__(self, cfg: dict):
        weights = Path(cfg.get("weights", "models/yolo11m.pt"))
        if not weights.is_absolute():
            weights = Path(__file__).resolve().parents[1] / weights
        weights.parent.mkdir(parents=True, exist_ok=True)

        # YOLO() downloads the checkpoint by name if the file is not present.
        self.model = YOLO(str(weights) if weights.exists() else weights.name)

        device = cfg.get("device", "auto")
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.imgsz = int(cfg.get("imgsz", 1280))
        self.conf = float(cfg.get("conf", 0.35))
        self.iou = float(cfg.get("iou", 0.5))
        self.tracker = cfg.get("tracker", "bytetrack.yaml")

    def reset(self) -> None:
        """Drop tracker state so IDs restart at 1 for the next camera."""
        for predictor_attr in ("predictor",):
            predictor = getattr(self.model, predictor_attr, None)
            if predictor is not None and hasattr(predictor, "trackers"):
                predictor.trackers = None
        self.model.predictor = None

    def update(self, frame: np.ndarray) -> list[Track]:
        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        out: list[Track] = []
        for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
            out.append(
                Track(
                    track_id=int(tid),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    conf=float(c),
                    foot=(float((x1 + x2) / 2), float(y2)),
                )
            )
        return out
