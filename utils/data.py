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

        box = [cls, conf] + box_scaler([x1, y1, x2, y2], origDims, targetDims)
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
        image_dim = data_i["imageDims"]
        prediction = data_i["textOfSample"]
        gt_plate_dim = get_plate_dim(labels_dir, image_id)

        cd_input_dim = data_i["cdInputDims"]
        cd_inference_time = data_i["detectionTime"]
        cd_boxes = sampleBoxes_to_predictions(
            data_i["cdSampleBoxes"], cd_input_dim, cd_input_dim
        )

        cr_input_dim = data_i["crInputDims"]
        cr_inference_time = data_i["recognitionTime"]

        print(f"Convertendo boxes {cr_input_dim} -> {gt_plate_dim}")
        
        cr_boxes = sampleBoxes_to_predictions(
            data_i["crSampleBoxes"], cr_input_dim, gt_plate_dim
        )

        predictions[image_id] = {
            "prediction": prediction,
            "cd_dim": cd_input_dim,
            "cr_dim": gt_plate_dim,
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


def scale_pred_boxes(groundtruths_dict: dict, predictions_dict) -> dict:
    
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
            "cr_dim": predictions_dict[image_id]["cr_dim"],
            "cd_boxes": predictions_dict[image_id]["cd_boxes"],
            "cr_boxes": cr_boxes,
            "cd_time": predictions_dict[image_id]["cd_time"],
            "cr_time": predictions_dict[image_id]["cr_time"],
        }

    return scaled_predictions


def plot_boxes(
    images_dir: Path,
    image_id: str,
    cd_input_dim: dict,
    cr_input_dim: dict,
    cd_boxes: list[float],
    cr_boxes: list[float],
):
    raw_image = cv.imread(str(images_dir / image_id))
    raw_image = cv.cvtColor(raw_image, cv.COLOR_BGR2RGB)

    W, H = cd_input_dim["width"], cd_input_dim["height"]
    w, h = cr_input_dim["width"], cr_input_dim["height"]

    print(W, H)

    image = cv.resize(raw_image, (W, H))

    for cd_box in cd_boxes:
        _, _, x1_cd, y1_cd, x2_cd, y2_cd = list(map(round, cd_box))

        image2plot = image.copy()
        plate_region = image2plot[y1_cd:y2_cd, x1_cd:x2_cd].copy()
        plate_region = cv.resize(plate_region, (w, h))

        print(plate_region.shape)

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(
            cv.rectangle(image2plot, (x1_cd, y1_cd), (x2_cd, y2_cd), (255, 0, 0), 3)
        )

        plt.subplot(1, 2, 2)
        for cr_box in cr_boxes:
            _, _, x1_cr, y1_cr, x2_cr, y2_cr = list(map(round, cr_box))
            plate_region = cv.rectangle(
                plate_region, (x1_cr, y1_cr), (x2_cr, y2_cr), (255, 0, 0), 1
            )

        plt.imshow(plate_region)
        plt.show()


if __name__ == "__main__":
    file = Path("/home/bolsista/dev/TCC/benchmark/labels/BCV3G65_cd.txt")
    gts = get_groundtruth(
        Path("/home/bolsista/dev/TCC/benchmark/images/"),
        Path("/home/bolsista/dev/TCC/benchmark/labels/"),
    )

    pprint(gts)
