"""Class taxonomies and label mappings used for cross-dataset evaluation.

COCO detectors predict COCO category ids (1..90 sparse ids, 80 used).
KITTI 2D labels use: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist,
Tram, Misc, DontCare.

To evaluate a COCO-pretrained detector on KITTI without retraining, both the
KITTI ground truth and the detector outputs are mapped onto a shared, smaller
"unified" taxonomy. Two presets are provided (named after the ngdet tutorial):

  driving3 : vehicle / person / cyclist       (coarse, primary metric)
  driving5 : car / truck / person / cyclist   (finer; Tram ignored)

Anything mapped to IGNORE on the GT side becomes an iscrowd=1 "ignore region"
(one copy per unified class), so detections falling on it are neither TP nor FP
and it never counts toward recall. Anything mapped to IGNORE on the detector
side is dropped.
"""

IGNORE = "__ignore__"

# COCO 2017 category id -> name (the 80 used ids out of 1..90)
COCO_ID2NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}
COCO_NAME2ID = {v: k for k, v in COCO_ID2NAME.items()}
# Ultralytics / "80-class contiguous" index -> COCO category id
COCO80_TO_91 = sorted(COCO_ID2NAME.keys())
COCO80_NAMES = [COCO_ID2NAME[i] for i in COCO80_TO_91]

KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting",
                 "Cyclist", "Tram", "Misc", "DontCare"]


TAXONOMIES = {
    "driving3": {
        "classes": ["vehicle", "person", "cyclist"],
        "kitti": {
            "Car": "vehicle", "Van": "vehicle", "Truck": "vehicle", "Tram": "vehicle",
            "Pedestrian": "person", "Person_sitting": "person",
            "Cyclist": "cyclist",
            "Misc": IGNORE, "DontCare": IGNORE,
        },
        "coco": {
            "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "train": "vehicle",
            "person": "person",
            "bicycle": "cyclist", "motorcycle": "cyclist",
        },
    },
    "driving5": {
        "classes": ["car", "truck", "person", "cyclist"],
        "kitti": {
            "Car": "car", "Van": "car",
            "Truck": "truck",
            "Pedestrian": "person", "Person_sitting": "person",
            "Cyclist": "cyclist",
            "Tram": IGNORE, "Misc": IGNORE, "DontCare": IGNORE,
        },
        "coco": {
            "car": "car",
            "truck": "truck", "bus": "truck",
            "person": "person",
            "bicycle": "cyclist", "motorcycle": "cyclist",
        },
    },
}


def unified_cat_id(taxonomy: str, name: str) -> int:
    """1-based category id of a unified class inside a taxonomy."""
    return TAXONOMIES[taxonomy]["classes"].index(name) + 1


def coco_id_to_unified(taxonomy: str, coco_cat_id: int):
    """Map a COCO category id to a unified category id, or None (drop)."""
    name = COCO_ID2NAME.get(coco_cat_id)
    target = TAXONOMIES[taxonomy]["coco"].get(name)
    if target is None or target == IGNORE:
        return None
    return unified_cat_id(taxonomy, target)
