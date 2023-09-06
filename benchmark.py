import argparse
from pathlib import Path

from utils.data import *
from utils.metrics import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Mobile OCR Benchmark")
    parser.add_argument("--images-dir", type=Path, help="Path to images directory")
    parser.add_argument("--labels-dir", type=Path, help="Path to labels directory")
    parser.add_argument("--json-path", type=Path, help="Path to json file")
    args = parser.parse_args()

    benchmark_dict = json_to_dict(args.json_path)
    predictions = get_predictions(args.labels_dir, benchmark_dict)
    groundtruths = get_groundtruth(args.images_dir, args.labels_dir)
    groundtruths = scale_gt_boxes(groundtruths, predictions)

    metrics = get_metrics(groundtruths, predictions)
    pprint(metrics)
