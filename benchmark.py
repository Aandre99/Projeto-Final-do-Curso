import argparse
from pathlib import Path

from utils.data import *
from utils.metrics import get_metrics
from utils.plot import plot_boxes, plot_two_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile OCR Benchmark")
    parser.add_argument("--images-dir", type=Path, help="Path to images directory")
    parser.add_argument("--labels-dir", type=Path, help="Path to labels directory")
    parser.add_argument("--json-path", type=Path, help="Path to json file")
    args = parser.parse_args()

    print("> Build json with Android embbeded model predictions\n")
    benchmark_dict = json_to_dict(args.json_path)

    print("> Build predictions dictionary\n")
    predictions = get_predictions(args.labels_dir, benchmark_dict)

    print("> Build groundtruths dictionary\n")
    groundtruths = get_groundtruth(args.images_dir, args.labels_dir)

    # plot_boxes(predictions, args.images_dir, n_samples=2)

    print("> Fit predictions boxes to groundtruths dimensions\n")
    predictions = fit_pred_boxes_to_gt_boxes(groundtruths, predictions)

    # plot_boxes(groundtruths, predictions, args.images_dir, n_samples=10)

    # plot_two_boxes(groundtruths,predictions, args.images_dir, n_samples=2)
    # plot_boxes(predictions, args.images_dir, n_samples=2)

    # print('> Computing metrics\n')
    metrics = get_metrics(groundtruths, predictions)
    pprint(metrics)
