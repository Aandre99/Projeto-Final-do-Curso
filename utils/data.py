import json
from pathlib import Path
from pprint import pprint

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

NAMES = {
    "cd": {0: "plate"},
    "cr": {k: v for k, v in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")},
}


def json_to_dict(json_path) -> list:
    data = None

    with open(json_path, "r") as f:
        data = json.load(f)

    if data is None:
        raise Exception("Error: json file is empty")
    return data


def box_scaler(box, origDims, targetDims):
    x1, y1, x2, y2 = box

    dx = targetDims["width"] / origDims["width"]
    dy = targetDims["height"] / origDims["height"]

    x1 = int(x1 * dx)
    y1 = int(y1 * dy)
    x2 = int(x2 * dx)
    y2 = int(y2 * dy)

    return [x1, y1, x2, y2]


def sampleBoxes_to_predictions(
    sample_boxes: list, origDims: tuple[int], targetDims: tuple[int]
) -> list:
    boxes = []

    for i in range(len(sample_boxes)):

        sample_box = sample_boxes[i]

        conf = sample_box["confidence"]
        cls = sample_box["detectedClass"]
        x1 = sample_box["location"]["left"]
        y1 = sample_box["location"]["top"]
        x2 = sample_box["location"]["right"]
        y2 = sample_box["location"]["bottom"]

        box = [cls, conf] + [x1, y1, x2, y2]
        boxes.append(box)

    return boxes


def get_plate_dim(labels_dir: Path, image_id: str):
    labels_path = labels_dir / (Path(image_id).stem + "_cr.txt")
    with open(labels_path, "r") as f:
        labels = f.readlines()
    w, h = labels[0].split(" ")[0].split(",")
    return {"width": int(w), "height": int(h)}


def get_predictions(labels_dir, data: list) -> dict:
    predictions = {}

    for i in range(len(data)):
        data_i = data[i]

        image_id = data_i["imageId"]
        prediction = data_i["textOfSample"]
        gt_plate_dim = get_plate_dim(labels_dir, image_id)

        cd_input_dim = data_i["cdInputDims"]
        cd_inference_time = data_i["detectionTime"]
        cd_boxes = sampleBoxes_to_predictions(
            data_i["cdSampleBoxes"], cd_input_dim, cd_input_dim
        )

        cr_input_dim = data_i["crInputDims"]
        cr_inference_time = data_i["recognitionTime"]

        cr_boxes = sampleBoxes_to_predictions(
            data_i["crSampleBoxes"], cr_input_dim, gt_plate_dim
        )

        predictions[image_id] = {
            "prediction": prediction,
            "cd_dim": cd_input_dim,
            "cr_dim": cr_input_dim,
            "cd_time": cd_inference_time,
            "cr_time": cr_inference_time,
            "cd_boxes": cd_boxes,
            "cr_boxes": cr_boxes,
        }

    return predictions


def load_detections_from_file(detections_path: Path) -> dict:
    info = {"label": "", "dims": [], "boxes": []}
    label = ""
    dims = []

    with open(detections_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            dims = list(map(int, line[0].split(",")))
            cls = int(line[2])
            label += line[1]
            box = [cls, 1.0] + list(map(float, line[3:]))
            info["boxes"].append(box)
            info["dims"] = dims

        info["label"] = label
        info["dims"] = {"width": dims[0], "height": dims[1]}

    return info


def get_groundtruth(images_dir: Path, labels_dir: Path) -> dict:
    groundtruth = {}

    for image_path in images_dir.rglob("*.[jJ][pP][gG]"):
        image_id = image_path.stem

        image = cv.imread(str(image_path))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        cr_label_path = labels_dir / (image_id + "_cr.txt")
        cd_label_path = labels_dir / (image_id + "_cd.txt")

        cd_info = load_detections_from_file(cd_label_path)
        cr_info = load_detections_from_file(cr_label_path)

        groundtruth[image_path.name] = {
            "groundtruth": cr_info["label"],
            "cd_dim": cd_info["dims"],
            "cr_dim": cr_info["dims"],
            "cd_boxes": cd_info["boxes"],
            "cr_boxes": cr_info["boxes"],
        }

    return groundtruth


def fit_pred_boxes_to_gt_boxes(groundtruths_dict: dict, predictions_dict) -> dict:
    scaled_predictions = {}

    pred_copy = predictions_dict.copy()

    for image_id, image_info in pred_copy.items():
        cr_boxes = []

        for cr_box in image_info["cr_boxes"]:
            cls, conf, x1, y1, x2, y2 = cr_box
            cr_boxes.append(
                [cls, conf]
                + box_scaler(
                    [x1, y1, x2, y2],
                    image_info["cr_dim"],
                    groundtruths_dict[image_id]["cr_dim"],
                )
            )

        scaled_predictions[image_id] = {
            "prediction": image_info["prediction"],
            "cd_dim": predictions_dict[image_id]["cd_dim"],
            "cr_dim": groundtruths_dict[image_id]["cr_dim"],
            "cd_boxes": predictions_dict[image_id]["cd_boxes"],
            "cr_boxes": cr_boxes,
            "cd_time": predictions_dict[image_id]["cd_time"],
            "cr_time": predictions_dict[image_id]["cr_time"],
        }

    return scaled_predictions


def compare_boxes(
    images_dir: Path,
    predictions: dict,
    groundtruths: dict,
    n_samples: int = 10,
):
    count = 0

    for image_id, image_info in groundtruths.items():
        if count == n_samples:
            break

        image = cv.imread(str(images_dir / image_id))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        for cd_box in image_info["cd_boxes"]:
            cls, conf, x1, y1, x2, y2 = cd_box
            plate = image[int(y1) : int(y2), int(x1) : int(x2)].copy()

            gt_cr_boxes = groundtruths[image_id]["cr_boxes"]
            pred_cr_boxes = predictions[image_id]["cr_boxes"]

            for gt_cr_box in gt_cr_boxes:
                cls, conf, x1, y1, x2, y2 = gt_cr_box

                plate = cv.rectangle(
                    plate, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1
                )

            for pred_cr_box in pred_cr_boxes:
                cls, conf, x1, y1, x2, y2 = pred_cr_box
                plate = cv.rectangle(
                    plate, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1
                )

            plt.imshow(plate)
            plt.show()
            count += 1


if __name__ == "__main__":
    file = Path("/home/bolsista/dev/TCC/benchmark/labels/BCV3G65_cd.txt")
    gts = get_groundtruth(
        Path("/home/bolsista/dev/TCC/benchmark/images/"),
        Path("/home/bolsista/dev/TCC/benchmark/labels/"),
    )

    pprint(gts)
