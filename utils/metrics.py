from collections import Counter
from pprint import pprint

import Levenshtein as lev
import numpy as np
from tqdm import tqdm


def get_levistain_distance(pred: str, gt: str) -> int:
    """
    This function calculates the Levenshtein distance between two strings.

    Args:
        pred (str): The predicted string.
        gt (str): The ground truth string.

    Returns:
        The Levenshtein distance between the two strings.
    """
    return lev.distance(pred, gt)


def filter_boxes_class(boxes: list, class_id: int) -> list:
    """
    This function filters the boxes by class.

    Args:
        boxes (list): A list of boxes.
        class_id (int): The class id.

    Returns:
        A list of boxes filtered by class.
    """

    filtered_boxes = []
    for detection in boxes:
        if detection[1] == class_id:
            filtered_boxes.append(detection)
    return filtered_boxes


def intersection_over_union(boxes_preds, boxes_labels):
    """
    This function calculates the intersection over union (IoU) between two boxes.

    Args:
        boxes_preds (list): A list of predicted boxes on [x1, y1, x2, y2] format.
        boxes_labels (list): A list of ground truth boxes on [x1, y1, x2, y2] format.

    Returns:
        The IoU between the two boxes.
    """

    if boxes_preds.ndim == 1:
        boxes_preds = boxes_preds.reshape(1, -1)
    if boxes_labels.ndim == 1:
        boxes_labels = boxes_labels.reshape(1, -1)

    box1_x1, box1_y1, box1_x2, box1_y2 = (
        boxes_preds[:, 0],
        boxes_preds[:, 1],
        boxes_preds[:, 2],
        boxes_preds[:, 3],
    )

    box2_x1, box2_y1, box2_x2, box2_y2 = (
        boxes_labels[:, 0],
        boxes_labels[:, 1],
        boxes_labels[:, 2],
        boxes_labels[:, 3],
    )

    x1 = np.maximum(box1_x1, box2_x1)
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box1_area = np.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = np.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

    return iou


def calculate_average_precision(precision, recall):
    """
    Calculates the average precision (AP) given precision and recall values.

    Args:
        precision (list): A list of precision values.
        recall (list): A list of recall values.

    Returns:
        The average precision (AP).
    """
    m_precision = np.concatenate(([0.0], precision, [0.0]))
    m_recall = np.concatenate(([0.0], recall, [1.0]))

    for i in range(len(m_precision) - 2, -1, -1):
        m_precision[i] = max(m_precision[i], m_precision[i + 1])

    indices = np.where(m_recall[1:] != m_recall[:-1])[0] + 1
    ap = np.sum((m_recall[indices] - m_recall[indices - 1]) * m_precision[indices])
    return ap


def compute_map(
    pred_boxes: list,
    true_boxes: list,
    iou_thresholds: list = [0.5],
    num_classes: int = 1,
) -> float:
    """
    Calculates mean average precision (mAP) for a range of IoU thresholds.

    Args:
    ----------
    pred_boxes (list): List of predicted bounding boxes.
    true_boxes (list): List of ground truth bounding boxes.
    iou_thresholds (list): List of IoU thresholds to compute mAP for.s
    num_classes (int): Number of classes.

    Returns:
        dict: mAP values for each IoU threshold.
    """
    mAPs = {}

    for iou_threshold in iou_thresholds:
        average_precisions = []
        epsilon = 1e-6

        for c in range(num_classes):
            
            detections = filter_boxes_class(pred_boxes, c)
            ground_truths = filter_boxes_class(true_boxes, c)
            
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            for key, val in amount_bboxes.items():
                amount_bboxes[key] = np.zeros(val)

            detections.sort(key=lambda x: x[2], reverse=True)

            TP = np.zeros((len(detections)))
            FP = np.zeros((len(detections)))

            total_true_bboxes = len(ground_truths)
            if total_true_bboxes == 0:
                continue
            
            for detection_idx, detection in enumerate(detections):
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                best_iou = 0
                best_gt_idx = -1

                for idx, gt in enumerate(ground_truth_img):
                    
                    iou = intersection_over_union(
                        np.array(detection[3:]), np.array(gt[3:])
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if (best_iou > iou_threshold) and (best_gt_idx != -1):
                    
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1

            TP_cumsum = np.cumsum(TP)
            FP_cumsum = np.cumsum(FP)
            
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            
            precisions = np.concatenate((np.array([1]), precisions))
            recalls = np.concatenate((np.array([0]), recalls))
            
            average_precisions.append(np.trapz(precisions, recalls))
            print(c, average_precisions[-1])
            
        mAPs[iou_threshold] = sum(average_precisions) / len(average_precisions)
        break
    return np.mean(list(mAPs.values()))


def get_metrics(groundtruths: dict, predictions: dict):
    """
    The function `get_metrics` calculates various metrics such as accuracy, average time, and mean
    average precision for a given set of groundtruths and predictions.

    Parameters
    ----------
    groundtruths : dict
        The `groundtruths` parameter is a dictionary that contains the ground truth information for each
    image. The keys of the dictionary are the image IDs, and the values are dictionaries that contain
    the ground truth information for that image.

    predictions : dict
        The `predictions` parameter is a dictionary that contains the predicted values for each image. The
    keys of the dictionary are the image IDs, and the values are dictionaries containing the predicted
    values for that image.

    Returns
    -------
        a dictionary containing various metrics such as accuracy, cd_time, cr_time, cd_map, and cr_map.

    """

    hits = np.zeros(len(groundtruths))
    cd_times = np.zeros(len(groundtruths))
    cr_times = np.zeros(len(groundtruths))

    cd_gt_boxes = []
    cd_pred_boxes = []

    cr_gt_boxes = []
    cr_pred_boxes = []

    lv_distances = []

    for i, image_id in enumerate(groundtruths.keys()):
        gt = groundtruths[image_id]
        pred = predictions[image_id]

        cd_times[i] = pred["cd_time"]
        cr_times[i] = pred["cr_time"]

        if pred["prediction"] == gt["groundtruth"]:
            hits[i] = 1
        else:
            print(f"{image_id} | {gt['groundtruth']} | {pred['prediction']}")

        lv_distances.append(
            get_levistain_distance(gt["groundtruth"], pred["prediction"])
        )

        cd_gt_box = list(map(lambda x: [i] + x, gt["cd_boxes"]))
        cd_pred_box = list(map(lambda x: [i] + x, pred["cd_boxes"]))
        cd_gt_boxes.extend(cd_gt_box)
        cd_pred_boxes.extend(cd_pred_box)

        cr_gt_box = list(map(lambda x: [i] + x, gt["cr_boxes"]))
        cr_pred_box = list(map(lambda x: [i] + x, pred["cr_boxes"]))
        cr_gt_boxes.extend(cr_gt_box)
        cr_pred_boxes.extend(cr_pred_box)

    accuracy = hits.mean() * 100
    cd_time = (cd_times.mean(), cd_times.std())
    cr_time = (cr_times.mean(), cr_times.std())
    cd_map = compute_map(cd_pred_boxes, cd_gt_boxes, np.arange(0.5, 1.0, 0.05), 1)
    cr_map = compute_map(cr_pred_boxes, cr_gt_boxes, np.arange(0.5, 1.0, 0.05), 36)
    lv_dist_mean = np.mean(lv_distances)

    metrics = {
        "accuracy": f"{accuracy:.2f}%",
        "CD Time": f"{cd_time[0]:.2f} ± {cd_time[1]:.2f} ms",
        "CR Time": f"{cr_time[0]:.2f} ± {cr_time[1]:.2f} ms",
        "CD mAP@0.5:0.95": f"{cd_map:.2f}",
        "CR mAP@0.5:0.95": f"{cr_map:.2f}",
        "lv_distance": f"{lv_dist_mean:.2f}",
    }

    return metrics
