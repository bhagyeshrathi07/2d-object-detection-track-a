"""Unified wrappers around the detector families under test.

Every wrapper exposes:
    .name            short id used in result files
    .family          detector family (two-stage / one-stage anchor-free / YOLO / DETR / open-vocab)
    .info()          dict with checkpoint, resolution, thresholds, framework version
    .predict(pil)    -> (boxes_xyxy [N,4] float, scores [N], coco_cat_ids [N] int)
                       boxes in original image pixel coordinates

All wrappers are run with a LOW score threshold (default 0.001) so that mAP,
which integrates over the full precision/recall curve, is not truncated.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from taxonomy import COCO80_NAMES, COCO80_TO_91, COCO_NAME2ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
#  torchvision: Faster R-CNN (two-stage), FCOS (anchor-free one-stage), RetinaNet
# --------------------------------------------------------------------------- #
class TorchvisionDetector:
    FAMILY = {
        "fasterrcnn_resnet50_fpn_v2": "Faster R-CNN (two-stage, anchor-based)",
        "fcos_resnet50_fpn": "FCOS (one-stage, anchor-free)",
        "retinanet_resnet50_fpn_v2": "RetinaNet (one-stage, anchor-based)",
    }

    def __init__(self, arch: str, score_thr: float = 0.001, nms_thr: float | None = None,
                 min_size: int = 800, max_size: int = 1333, fp16: bool = False):
        import torchvision
        from torchvision.models import detection as D

        self.name = arch
        self.family = self.FAMILY[arch]
        self.arch = arch
        self.fp16 = fp16
        weights_enum = {
            "fasterrcnn_resnet50_fpn_v2": D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            "fcos_resnet50_fpn": D.FCOS_ResNet50_FPN_Weights.COCO_V1,
            "retinanet_resnet50_fpn_v2": D.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        }[arch]
        self.checkpoint = str(weights_enum)
        kw = dict(weights=weights_enum, box_score_thresh=score_thr if arch.startswith("fasterrcnn") else None,
                  score_thresh=None if arch.startswith("fasterrcnn") else score_thr,
                  min_size=min_size, max_size=max_size)
        kw = {k: v for k, v in kw.items() if v is not None}
        if nms_thr is not None:
            kw["box_nms_thresh" if arch.startswith("fasterrcnn") else "nms_thresh"] = nms_thr
        self.model = getattr(D, arch)(**kw).to(DEVICE).eval()
        # record the effective thresholds
        if arch.startswith("fasterrcnn"):
            self.score_thr = self.model.roi_heads.score_thresh
            self.nms_thr = self.model.roi_heads.nms_thresh
            self.max_det = self.model.roi_heads.detections_per_img
        else:
            self.score_thr = self.model.score_thresh
            self.nms_thr = self.model.nms_thresh
            self.max_det = self.model.detections_per_img
        self.min_size, self.max_size = min_size, max_size
        self.framework = f"torchvision {torchvision.__version__} / torch {torch.__version__}"
        self.weights = weights_enum

    def parameters(self):
        return self.model.parameters()

    def info(self):
        return dict(name=self.name, family=self.family, framework=self.framework,
                    checkpoint=self.checkpoint, input_resolution=f"shorter side {self.min_size}, longer side <= {self.max_size} (aspect preserved)",
                    score_thr=self.score_thr, nms_thr=self.nms_thr, max_det=self.max_det,
                    precision="fp16 autocast" if self.fp16 else "fp32")

    @torch.inference_mode()
    def predict(self, img: Image.Image):
        from torchvision.transforms.functional import to_tensor
        x = to_tensor(img).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model([x])[0]
        return (out["boxes"].float().cpu().numpy(), out["scores"].float().cpu().numpy(),
                out["labels"].cpu().numpy().astype(int))


# --------------------------------------------------------------------------- #
#  Ultralytics: YOLO11, RT-DETR, YOLO-World
# --------------------------------------------------------------------------- #
class UltralyticsDetector:
    def __init__(self, weights: str, score_thr: float = 0.001, nms_thr: float = 0.7,
                 imgsz: int = 640, fp16: bool = False, max_det: int = 300,
                 vocab: list[str] | None = None, vocab_to_coco: dict[str, str] | None = None):
        import ultralytics
        from ultralytics import YOLO, RTDETR, YOLOWorld

        self.weights = weights
        self.name = weights.replace(".pt", "")
        self.score_thr, self.nms_thr, self.imgsz, self.fp16, self.max_det = score_thr, nms_thr, imgsz, fp16, max_det
        self.framework = f"ultralytics {ultralytics.__version__} / torch {torch.__version__}"
        self.vocab = None
        if "rtdetr" in weights:
            self.model = RTDETR(weights)
            self.family = "RT-DETR (real-time DETR, transformer, NMS-free)"
            self.nms_thr = None
        elif "world" in weights:
            self.model = YOLOWorld(weights)
            self.family = "YOLO-World (open-vocabulary YOLO)"
            vocab = vocab or COCO80_NAMES
            self.model.set_classes(vocab)
            self.vocab = vocab
            self.name = self.name + ("" if vocab == COCO80_NAMES else "_customvocab")
        else:
            self.model = YOLO(weights)
            self.family = "YOLO11 (one-stage, anchor-free)"
        # vocab index -> COCO category id
        if self.vocab is not None:
            vocab_to_coco = vocab_to_coco or {}
            self.idx2coco = np.array([COCO_NAME2ID[vocab_to_coco.get(v, v)] for v in self.vocab])
        else:
            self.idx2coco = np.array(COCO80_TO_91)

    def parameters(self):
        return self.model.model.parameters()

    def info(self):
        return dict(name=self.name, family=self.family, framework=self.framework,
                    checkpoint=self.weights, input_resolution=f"letterboxed to {self.imgsz} (longest side)",
                    score_thr=self.score_thr, nms_thr=self.nms_thr, max_det=self.max_det,
                    precision="fp16" if self.fp16 else "fp32", vocab=self.vocab)

    def predict(self, img: Image.Image):
        kw = dict(conf=self.score_thr, imgsz=self.imgsz, half=self.fp16, max_det=self.max_det,
                  verbose=False, device=DEVICE)
        if self.nms_thr is not None:
            kw["iou"] = self.nms_thr
        r = self.model.predict(img, **kw)[0]
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int)
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), self.idx2coco[cls]


# --------------------------------------------------------------------------- #
#  registry
# --------------------------------------------------------------------------- #
KITTI_PROMPTS = ["car", "van", "truck", "pedestrian", "cyclist", "tram"]
# prompt -> COCO name, so the generic taxonomy mapping can be reused downstream
KITTI_PROMPT_TO_COCO = {"car": "car", "van": "car", "truck": "truck",
                        "pedestrian": "person", "cyclist": "bicycle", "tram": "train"}
# variant 2: same, but the COCO word "person" instead of KITTI's "pedestrian"
KITTI_PROMPTS2 = ["car", "van", "truck", "person", "cyclist", "tram"]
KITTI_PROMPT2_TO_COCO = {"car": "car", "van": "car", "truck": "truck",
                         "person": "person", "cyclist": "bicycle", "tram": "train"}


def build_detector(spec: str, score_thr: float = 0.001, fp16: bool = False, imgsz: int = 640):
    """spec examples:
        tv:fasterrcnn_resnet50_fpn_v2   tv:fcos_resnet50_fpn   tv:retinanet_resnet50_fpn_v2
        ul:yolo11n.pt  ul:yolo11m.pt  ul:rtdetr-l.pt  ul:yolov8s-worldv2.pt
        ulkitti:yolov8s-worldv2.pt   (YOLO-World prompted with KITTI class names)
        ulkitti2:yolov8s-worldv2.pt  (same prompts but 'person' instead of 'pedestrian')
    """
    kind, _, arg = spec.partition(":")
    if kind == "tv":
        return TorchvisionDetector(arg, score_thr=score_thr, fp16=fp16)
    if kind == "ul":
        return UltralyticsDetector(arg, score_thr=score_thr, fp16=fp16, imgsz=imgsz)
    if kind == "ulkitti":
        return UltralyticsDetector(arg, score_thr=score_thr, fp16=fp16, imgsz=imgsz,
                                   vocab=KITTI_PROMPTS, vocab_to_coco=KITTI_PROMPT_TO_COCO)
    if kind == "ulkitti2":
        return UltralyticsDetector(arg, score_thr=score_thr, fp16=fp16, imgsz=imgsz,
                                   vocab=KITTI_PROMPTS2, vocab_to_coco=KITTI_PROMPT2_TO_COCO)
    raise ValueError(spec)


def cuda_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timed_predict(det, img):
    cuda_sync()
    t0 = time.perf_counter()
    out = det.predict(img)
    cuda_sync()
    return out, (time.perf_counter() - t0) * 1000.0
